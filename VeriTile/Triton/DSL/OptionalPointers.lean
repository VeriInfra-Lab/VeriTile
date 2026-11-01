import VeriTile.Triton.DSL.PythonInlineSuites

/-! Caller-side presence of nullable pointer arguments. The pasted body retains
Python's `pointer is None` / `pointer is not None` guards. Every use of a
nullable pointer must be dominated by its presence guard, and rebinding is
rejected. Presence is frozen before source constexpr names are introduced. -/
namespace VeriTile.Triton.DSL.OptionalPointers
open Lean Elab Term Meta

syntax (name := withCallPresence) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " optional_pointers " "[" tritonCallBinding,* "]"
  " in " "triton " RawSourceText.body : term
syntax (name := withSignaturePresence) "with_triton_signature " term
  " optional_pointers " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "pointer argument profile must have statically known names"

private def pointerNames (profile : Expr) : TermElabM (List Name) := do
  let mut names := []
  for field in [``Triton.CallSignature.pointers, ``Triton.CallSignature.floatingPointers] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      let name ← whnf (← mkAppM ``Prod.fst #[pair])
      let .lit (.strVal name) := name | throwError "pointer parameter name must be literal"
      names := names ++ [Name.mkSimple name]
  return names

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers

/-- Mutated pointer arguments need an SSA binding, rather than repeatedly
adding the entry offset. Until that binding is supported, refuse such bodies. -/
private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some index := args.findIdx? (fun a => a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
      for id in args.take index do
        if id.isIdent && names.contains id.getId.eraseMacroScopes then
          throwErrorAt id "optional_pointers currently requires an immutable pointer parameter"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwErrorAt counter "a loop cannot shadow an optional pointer parameter"
  for arg in raw.getArgs do rejectWrites names arg

syntax (name := isNotNone) ident " is " "not " "None" : tritonExpr
syntax (name := isNone) ident " is " "None" : tritonExpr

private def guard? (raw : Syntax) : Option (Name × Bool) :=
  match raw with
  | `(tritonExpr| $id:ident is not None) => some (id.getId.eraseMacroScopes,Bool.true)
  | `(tritonExpr| $id:ident is None) => some (id.getId.eraseMacroScopes,Bool.false)
  | _ => none

private partial def substitute (values : List (Name × Ident)) (present : List Name)
    (raw : Syntax) : MacroM Syntax := do
  if let some (name,positive) := guard? raw then
    let some (_, bound) := values.find? (fun p => p.1 == name)
      | Macro.throwErrorAt raw "None guard requires a reviewed optional pointer parameter"
    if positive then return (← `(tritonExpr| $bound:ident)).raw
    else return (← `(tritonExpr| not $bound:ident)).raw
  let branch (condition : TSyntax `tritonExpr) (yes no : Array (TSyntax `tritonStmt)) (hasElse : Bool) : MacroM Syntax := do
    let known := (guard? condition).filter (fun p => values.any (fun v => v.1 == p.1))
    let yesKnown := match known with | some (name,Bool.true) => name::present | _ => present
    let noKnown := match known with | some (name,Bool.false) => name::present | _ => present
    let condition : TSyntax `tritonExpr := ⟨← substitute values present condition⟩
    let yes : Array (TSyntax `tritonStmt) ← yes.mapM fun s => return ⟨← substitute values yesKnown s⟩
    let no : Array (TSyntax `tritonStmt) ← no.mapM fun s => return ⟨← substitute values noKnown s⟩
    if hasElse then return (← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* })).raw
    else return (← `(tritonStmt| if $condition:tritonExpr { $yes* })).raw
  if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) := raw then
    return ← branch condition yes no Bool.true
  if let `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) := raw then
    return ← branch condition yes #[] Bool.false
  if let `(tritonExpr| $id:ident) := raw then
    let name := id.getId.eraseMacroScopes
    if values.any (fun p => p.1 == name) && !present.contains name then
      Macro.throwErrorAt id "optional pointer use requires a dominating presence guard"
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (substitute values present))
  | other => return other

/-- Separate tokens explicitly; the default formatter for a repeated custom
statement category need not insert spaces or newlines between statements. -/
private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args =>
    if kind==choiceKind then tokens args[0]!
    else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out := out++" "
    out := out++token
    previous := token
  return out

private def elaborate (stx : Syntax) (hasScalars : Bool) (expected : Option Expr) : TermElabM Expr := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let pointers ← pointerNames profile
  let offsets := stx[if hasScalars then 8 else 4].getSepArgs
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body
    | .error message => throwErrorAt raw "{message}"
  let mut bindings : List (Name × Expr) := []
  for binding in offsets do
    let `(tritonCallBinding| $id:ident := $value:term) := binding
      | throwErrorAt binding "expected a named optional pointer presence"
    let name := id.getId.eraseMacroScopes
    unless pointers.contains name do throwErrorAt id "optional binding is not a pointer parameter"
    if bindings.any (fun p => p.1 == name) then throwErrorAt id "duplicate optional pointer binding"
    let value ← elabTermEnsuringType value (mkConst ``Bool)
    bindings := bindings ++ [(name,value)]
  rejectWrites (bindings.map Prod.fst) parsed
  let used := identifiers parsed ++ identifiers stx
  -- Freeze caller expressions before source constexpr lets are introduced.
  -- Fresh names absent from the body prevent accidental source-name capture.
  let rec bind (todo : List (Name × Expr)) (values : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let mut localName := Name.anonymous
      repeat
        let fresh ← mkFreshId
        localName := Name.mkSimple ("veritile_pointer_presence_" ++
          (fresh.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains localName do break
      withLetDecl localName (mkConst ``Bool) value fun bound => do
        let result ← bind tail (values ++ [(name,mkIdent localName)])
        mkLetFVars #[bound] result
    | [] =>
      let body ← liftMacroM <| substitute values [] parsed
      let `(triton { $stmts:tritonStmt* }) := body
        | throwError "expected a complete pointer-argument body"
      let text := "{\n" ++ String.intercalate "\n" (stmts.toList.map (render ·.raw)) ++ "\n}"
      let raw := Syntax.atom .none text
      let base : Syntax := if hasScalars then
        mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],raw]
        else mkNode ``CallSignature.withSignature #[stx[0],stx[1],stx[6],stx[7],raw]
      if hasScalars then PythonInlineSuites.elaborateInvocation base expected
      else PythonInlineSuites.elaborateSignature base expected
  bind bindings []

@[term_elab withCallPresence] def elaborateInvocation : TermElab := fun stx expected =>
  elaborate stx Bool.true expected
@[term_elab withSignaturePresence] def elaborateSignature : TermElab := fun stx expected =>
  elaborate stx Bool.false expected

end VeriTile.Triton.DSL.OptionalPointers

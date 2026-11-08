import VeriTile.Triton.DSL.PointerArguments
import VeriTile.Triton.DSL.PythonFullConstructors

/-! Caller offsets for immutable pointers and one top-level self-add binding.
The caller base and first source displacement are added as signed values before
the final memory-coordinate conversion. Original assignments remain present.
Subsequent writes and control-flow bindings are refused rather than guessed. -/
namespace VeriTile.Triton.DSL.SingleAssignmentPointerOffsets
open Lean Elab Term Meta

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

/-- Only one top-level self-add may bind a caller pointer. Other mutations
are refused until their control-flow/value joins have an explicit lowering. -/
private partial def writes (name : Name) (raw : Syntax) : Bool :=
  let args:=raw.getArgs.toList
  let assignment:=["=",":=","+=","-=","*=","/="].any fun operator=>
    match args.findIdx? (fun a=>a.isAtom && a.getAtomVal.trimAscii.toString==operator) with
    | some i => (args.take i).any (fun a=>a.isIdent && a.getId.eraseMacroScopes==name.eraseMacroScopes)
    | none => Bool.false
  let loop:=match args with
    | a::b::_ => a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
      b.isIdent && b.getId.eraseMacroScopes==name.eraseMacroScopes
    | _ => Bool.false
  assignment || loop || raw.getArgs.any (writes name)

private partial def substitute (values : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_, offset) := values.find? (fun p => p.1 == id.getId.eraseMacroScopes) then
      return (← `(tritonExpr| ($id:ident + $($offset:ident)))).raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (substitute values))
  | other => return other

private def lowerBody (values : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  let `(triton { $stmts:tritonStmt* }):=raw | Macro.throwError "expected a complete source body"
  let mut active:=values
  let mut bound : List Name:=[]
  let mut out:=#[]
  for stmt in stmts do
    for name in bound do
      if writes name stmt.raw then Macro.throwError "an offset pointer may be assigned only once"
    let selfAdd:=match stmt with
      | `(tritonStmt| $id:ident = $base:ident + $offset:tritonExpr) =>
          if id.getId.eraseMacroScopes==base.getId.eraseMacroScopes then some (id,offset) else none
      | `(tritonStmt| $id:ident += $offset:tritonExpr) => some (id,offset)
      | _ => none
    if let some (id,offset):=selfAdd then
      if let some (_,base):=active.find? (fun p=>p.1==id.getId.eraseMacroScopes) then
        let offset : TSyntax `tritonExpr := ⟨← substitute active offset.raw⟩
        let next ← `(tritonStmt| $id:ident = ($id:ident + $($base:ident)) + $offset:tritonExpr)
        out:=out.push next
        active:=active.filter (fun p=>p.1!=id.getId.eraseMacroScopes)
        bound:=bound++[id.getId.eraseMacroScopes]
        continue
    for (name,_) in active do
      if writes name stmt.raw then Macro.throwError "an offset pointer requires one top-level self-add binding"
    out:=out.push (⟨← substitute active stmt.raw⟩ : TSyntax `tritonStmt)
  `(triton { $out:tritonStmt* })

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
      | throwErrorAt binding "expected a named pointer offset"
    let name := id.getId.eraseMacroScopes
    unless pointers.contains name do throwErrorAt id "offset binding is not a pointer parameter"
    if bindings.any (fun p => p.1 == name) then throwErrorAt id "duplicate pointer offset binding"
    let value ← elabTermEnsuringType value (mkConst ``Nat)
    bindings := bindings ++ [(name,value)]
  let used := identifiers parsed ++ identifiers stx
  -- Freeze caller expressions before source constexpr lets are introduced.
  -- Fresh names absent from the body prevent accidental source-name capture.
  let rec bind (todo : List (Name × Expr)) (values : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let mut localName := Name.anonymous
      repeat
        let fresh ← mkFreshId
        localName := Name.mkSimple ("veritile_pointer_offset_" ++
          (fresh.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains localName do break
      withLetDecl localName (mkConst ``Nat) value fun bound => do
        let result ← bind tail (values ++ [(name,mkIdent localName)])
        mkLetFVars #[bound] result
    | [] =>
      let body ← liftMacroM <| lowerBody values parsed
      let `(triton { $stmts:tritonStmt* }) := body
        | throwError "expected a complete pointer-argument body"
      let text := "{\n" ++ String.intercalate "\n" (stmts.toList.map (render ·.raw)) ++ "\n}"
      let raw := Syntax.atom .none text
      let base : Syntax := if hasScalars then
        mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[10],stx[11],raw]
        else mkNode ``CallSignature.withSignature #[stx[0],stx[1],stx[6],stx[7],raw]
      if hasScalars then PythonFullConstructors.elaborateInvocation base expected
      else PythonFullConstructors.elaborateSignature base expected
  bind bindings []

@[term_elab PointerArguments.withCallOffsets] def elaborateInvocation : TermElab := fun stx expected =>
  elaborate stx Bool.true expected
@[term_elab PointerArguments.withSignatureOffsets] def elaborateSignature : TermElab := fun stx expected =>
  elaborate stx Bool.false expected

end VeriTile.Triton.DSL.SingleAssignmentPointerOffsets

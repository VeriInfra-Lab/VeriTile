import VeriTile.Triton.DSL.PythonInlineSuites

/-! Caller-side cell coordinates for a pasted element-addressed kernel.
Each immutable pointer is viewed in a memory whose element i is stored at
base + cellWidth * i. This describes the memory representation; it does not
insert source statements or change arithmetic inside the pasted body.
The supported form is a direct pointer plus one complete natural displacement.
Other uses are rejected rather than silently mixing coordinate systems. -/
namespace VeriTile.Triton.DSL.PointerCells
open Lean Elab Term Meta

syntax (name := withSignatureCells) "with_triton_signature " term
  " pointer_cells " "[" tritonCallBinding,* "]" " in " "triton " RawSourceText.body : term

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
          throwErrorAt id "pointer_cells currently requires an immutable pointer parameter"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwErrorAt counter "a loop cannot shadow an cell-view pointer parameter"
  for arg in raw.getArgs do rejectWrites names arg

private partial def substitute (values : List (Name × Ident × Ident)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $id:ident + $offset:tritonExpr) := raw then
    if let some (_, base, width) := values.find? (fun p => p.1 == id.getId.eraseMacroScopes) then
      let offset : TSyntax `tritonExpr := ⟨← substitute values offset⟩
      return (← `(tritonExpr| $id:ident + ($($base:ident) + $($width:ident) * $offset:tritonExpr))).raw
  if let `(tritonExpr| $left:tritonExpr + $_:tritonExpr) := raw then
    if (identifiers left).any (fun id => values.any (fun p => p.1 == id)) then
      Macro.throwErrorAt left "pointer_cells requires one parenthesized complete displacement"
  if let `(tritonExpr| $id:ident) := raw then
    if values.any (fun p => p.1 == id.getId.eraseMacroScopes) then
      Macro.throwErrorAt id "pointer_cells requires a direct pointer plus a complete natural displacement"
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (substitute values))
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

private def elaborate (stx : Syntax) (expected : Option Expr) : TermElabM Expr := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let pointers ← pointerNames profile
  let offsets := stx[4].getSepArgs
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body
    | .error message => throwErrorAt raw "{message}"
  let mut bindings : List (Name × Expr × Expr) := []
  for binding in offsets do
    let `(tritonCallBinding| $id:ident := $value:term) := binding
      | throwErrorAt binding "expected a named pointer cell view"
    let name := id.getId.eraseMacroScopes
    unless pointers.contains name do throwErrorAt id "cell-view binding is not a pointer parameter"
    if bindings.any (fun p => p.1 == name) then throwErrorAt id "duplicate pointer cell-view binding"
    let pair ← elabTermEnsuringType value (mkApp2 (mkConst ``Prod [Level.zero,Level.zero]) (mkConst ``Nat) (mkConst ``Nat))
    let base ← mkAppM ``Prod.fst #[pair]
    let width ← mkAppM ``Prod.snd #[pair]
    bindings := bindings ++ [(name,base,width)]
  rejectWrites (bindings.map Prod.fst) parsed
  let used := identifiers parsed ++ identifiers stx
  -- Freeze caller expressions before source constexpr lets are introduced.
  -- Fresh names absent from the body prevent accidental source-name capture.
  let rec bind (todo : List (Name × Expr × Expr)) (values : List (Name × Ident × Ident)) : TermElabM Expr := do
    match todo with
    | (name,base,width)::tail =>
      let mut localName := Name.anonymous
      repeat
        let fresh ← mkFreshId
        localName := Name.mkSimple ("veritile_pointer_cell_" ++
          (fresh.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains localName || used.contains (Name.mkSimple (localName.toString ++ "_width")) do break
      let widthName := Name.mkSimple (localName.toString ++ "_width")
      withLetDecl localName (mkConst ``Nat) base fun bound => do
        withLetDecl widthName (mkConst ``Nat) width fun widthBound => do
          let result ← bind tail (values ++ [(name,mkIdent localName,mkIdent widthName)])
          mkLetFVars #[bound,widthBound] result
    | [] =>
      let body ← liftMacroM <| substitute values parsed
      let `(triton { $stmts:tritonStmt* }) := body
        | throwError "expected a complete pointer-argument body"
      let text := "{\n" ++ String.intercalate "\n" (stmts.toList.map (render ·.raw)) ++ "\n}"
      let raw := Syntax.atom .none text
      let base := mkNode ``CallSignature.withSignature #[stx[0],stx[1],stx[6],stx[7],raw]
      PythonInlineSuites.elaborateSignature base expected
  bind bindings []

@[term_elab withSignatureCells] def elaborateSignature : TermElab := fun stx expected =>
  elaborate stx expected

end VeriTile.Triton.DSL.PointerCells

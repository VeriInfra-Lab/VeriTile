import VeriTile.Triton.DSL.PointerLiteralOffsets

/-! Python floating literal aliases are compile-time scalars. Substitute their
literal syntax at uses so the ordinary expression typer retains the consuming
tensor's precision. The original assignment remains in the IR and pasted body.
Only single-write top-level literal bindings are substituted; mutable bindings
and nonliteral arithmetic are left to the ordinary checked elaborator. -/
namespace VeriTile.Triton.DSL.ReadonlyFloatAliases
open Lean Elab Term Meta

private partial def choose (raw : Syntax) : Syntax :=
  if raw.getKind==choiceKind then choose raw[0]!
  else raw.setArgs (raw.getArgs.map choose)
private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def writes (raw : Syntax) : List Name := Id.run do
  let args:=raw.getArgs.toList
  let mut names:=args.flatMap writes
  let leaf:=match raw.getKind with | .str _ name=>name | _=>""
  -- Keyword arguments are not assignments to local variables.
  if leaf.startsWith "tritonStmt" then
    for operator in ["=",":=","+=","-=","*=","/=","//=","%=","&=","|=","^="] do
      if let some i:=args.findIdx? (fun a=>a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
        names:=names++(args.take i).flatMap ids
  if let first::counter::_:=args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString && counter.isIdent then
      names:=counter.getId.eraseMacroScopes::names
  return names
private partial def literal (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| $_:scientific)=>Bool.true
  | `(tritonExpr| ($value:tritonExpr))=>literal value
  | `(tritonExpr| -$value:tritonExpr)=>literal value
  | _=>Bool.false
private partial def substitute (aliases : List (Name × TSyntax `tritonExpr))
    (raw : Syntax) : MacroM Syntax := do
  if let `(tritonExpr| $name:ident):=raw then
    if let some (_,value):=aliases.find? (·.1==name.getId.eraseMacroScopes) then
      return (← `(tritonExpr| ($value:tritonExpr))).raw
  if raw.isIdent then
    if let some (_,value):=aliases.find? (·.1==raw.getId.eraseMacroScopes) then
      return (← `(tritonExpr| ($value:tritonExpr))).raw
  let args:=raw.getArgs
  let leaf:=match raw.getKind with | .str _ name=>name | _=>""
  let keywordOwner:=leaf.startsWith "tritonExpr" || leaf.startsWith "tritonMemKwarg" ||
    leaf.startsWith "tritonReduceKwarg"
  let mut lowered:=#[]
  for i in [:args.size] do
    let key:=keywordOwner && args[i]!.isIdent && args[i+1]?.any
      (fun token=>token.isAtom && token.getAtomVal.trimAscii.toString=="=")
    let next ← if key then pure args[i]! else substitute aliases args[i]!
    lowered:=lowered.push next
  return raw.setArgs lowered

/-- A literal binding is retained, while its immutable uses get the same
contextual typing as an inline Python scalar. No future binding is consulted. -/
def lower (body : Array (TSyntax `tritonStmt)) : MacroM (Array (TSyntax `tritonStmt)) := do
  let body:=body.map fun st=>(⟨choose st.raw⟩ : TSyntax `tritonStmt)
  let assigned:=body.toList.flatMap (writes ·.raw)
  let mut aliases:List (Name × TSyntax `tritonExpr):=[]
  let mut out:=#[]
  for st in body do
    let next:TSyntax `tritonStmt:=⟨← substitute aliases st.raw⟩
    if let `(tritonStmt| $name:ident = $value:tritonExpr):=next then
      let key:=name.getId.eraseMacroScopes
      if assigned.count key==1 && literal value then aliases:=aliases++[(key,value)]
    out:=out.push next
  return out

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "literal alias call metadata requires static lists"
private def prepare (stx : Syntax) : TermElabM Syntax := do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let mut pointers:=[]
  for projection in [``Triton.CallSignature.pointers,``Triton.CallSignature.floatingPointers] do
    for pair in ← listValues (← mkAppM projection #[profile]) do
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair]) | throwError "pointer names must be literal"
      pointers:=pointers++[name.toName]
  let stx ← PythonFullConstructors.prepareSource stx
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok value=>pure value | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok value=>pure (choose value) | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete source required"
  let body ← liftMacroM (lower body)
  let body ← liftMacroM (PointerLiteralOffsets.lower pointers body)
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n"
    (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)
@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  StaticRanges.elaborateInvocation (← prepare stx) expected
end VeriTile.Triton.DSL.ReadonlyFloatAliases

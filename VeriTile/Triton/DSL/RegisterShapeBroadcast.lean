import VeriTile.Triton.DSL.IntegerDTypeCallConstants
import VeriTile.Triton.DSL.BooleanCallControl

/-! Attribute-shaped broadcasts use the actual shape inferred for the receiver
by the existing typed elaborator. No independent shape metadata is accepted.
Only a preceding, unconditional assignment can establish this shape. -/
namespace VeriTile.Triton.DSL.RegisterShapeBroadcast
open Lean Elab Term Meta
syntax (name := registerBroadcast) "tl.broadcast_to(" tritonExpr ", " ident ")" : tritonExpr
private partial def mentionsShape (raw : Syntax) : Bool :=
  raw.getKind==``registerBroadcast || raw.getArgs.any mentionsShape
private partial def writes (name : Name) (raw : Syntax) : Bool :=
  let own:=match (⟨raw⟩ : TSyntax `tritonStmt) with
    | `(tritonStmt| $id:ident = $_:tritonExpr)
    | `(tritonStmt| $id:ident += $_:tritonExpr)
    | `(tritonStmt| $id:ident -= $_:tritonExpr)
    | `(tritonStmt| $id:ident *= $_:tritonExpr)
    | `(tritonStmt| $id:ident /= $_:tritonExpr)=>id.getId.eraseMacroScopes==name
    | _=>Bool.false
  own || raw.getArgs.any (writes name)
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "broadcast receiver requires a statically elaborated source prefix"
private def receiverShape (before : Array (TSyntax `tritonStmt)) (name : Name) : TermElabM (Array (TSyntax `tritonExpr)) := do
  let mut found:=Bool.false
  for stmt in before do
    if let `(tritonStmt| $id:ident = $_:tritonExpr):=stmt then
      if id.getId.eraseMacroScopes==name then found:=Bool.true
    else if writes name stmt then
      throwError "broadcast shape receiver must have an unconditional assignment"
  unless found do throwError "broadcast shape receiver must have a preceding assignment"
  liftMacroM do
    let statements:=before.toList.filter fun stmt=>match stmt.raw with
      | `(tritonStmt| tl.ports inputs=[$_:term,*], outputs=[$_:term,*])=>Bool.false
      | _=>Bool.true
    let statements ← statements.mapM fun stmt=>do
      return (⟨← Inference.normalizeControlFlow stmt.raw⟩ : TSyntax `tritonStmt)
    unless statements.any (fun st=>st.raw.getKind==``tritonCallSignature) do
      Macro.throwError "register.shape requires typed call metadata"
    let regions:=Inference.collectRegionDTypes statements
    let pointers:=Inference.collectPtrElems regions statements
    let context : FPSourceContext:={regions:=regions,callSignature:=Bool.true,legacyPointerCasts:=Bool.false}
    let (_,_,env,_) ← expandStmts [] (Inference.collectNatPinned statements)
      (Inference.collectIntPinned statements) regions pointers context [] [] statements
    let (_, .dims dims) ← lookupEnv env name.toString
    dims.toArray.mapM fun dim=>do
      let `(term| ($value:term : Nat)):=dim
        | Macro.throwError "receiver shape dimension requires a canonical Nat expression"
      `(tritonExpr| $($value))

private partial def lower (before : Array (TSyntax `tritonStmt)) (raw : Syntax) : TermElabM Syntax := do
  if raw.getKind==choiceKind then
    let selected:=raw.getArgs.find? fun child=>match child with
      | `(tritonStmt| for $_:ident in range($($_:term), $($_:term), $($_:term)) { $_:tritonStmt* })=>Bool.true
      | _=>Bool.false
    return ← lower before (selected.getD raw[0]!)
  if raw.getKind==``registerBroadcast then
    let `(tritonExpr| tl.broadcast_to($value:tritonExpr, $attr:ident)):=raw
      | throwError "expected register.shape broadcast target"
    let .str receiver "shape":=attr.getId.eraseMacroScopes
      | throwError "broadcast target must be register.shape"
    let dims ← receiverShape before receiver
    let value : TSyntax `tritonExpr:=⟨← lower before value⟩
    return (← liftMacroM `(tritonExpr| tl.broadcast_to($value:tritonExpr, [$dims,*]))).raw
  return raw.setArgs (← raw.getArgs.mapM (lower before))
def prepare (stx : Syntax) : TermElabM Syntax := do
  unless mentionsShape stx do return stx
  let `(triton { $body:tritonStmt* }):=stx | throwUnsupportedSyntax
  let mut before:=#[]
  for stmt in body do
    let stmt ← lower before stmt
    before:=before.push (⟨stmt⟩ : TSyntax `tritonStmt)
  return (← liftMacroM `(triton { $before* })).raw
@[term_elab tritonBlock] def elaborateBroadcast : TermElab := fun stx expected=>do
  let next ← prepare stx
  if next==stx then throwUnsupportedSyntax
  elabTerm next expected

private partial def signedZeroStop (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM signedZeroStop)
  if let `(tritonStmt| for $i:ident in range($start:tritonExpr, 0, -$step:tritonExpr) { $body:tritonStmt* }):=raw then
    return (← `(tritonStmt| for $i:ident in range($start:tritonExpr, -0, -$step:tritonExpr) { $body* })).raw
  return raw
@[term_elab CallSignature.withCall] def elaborateInvocation : TermElab := fun stx expected=>do
  let stx ← PythonFullConstructors.prepareSource stx
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwError "{message}"
  let `(triton { $body:tritonStmt* }):= ← liftMacroM (signedZeroStop parsed)
    | throwError "complete body required"
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let adjusted:=stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)
  IntegerDTypeCallConstants.elaborateInvocation adjusted expected
end VeriTile.Triton.DSL.RegisterShapeBroadcast

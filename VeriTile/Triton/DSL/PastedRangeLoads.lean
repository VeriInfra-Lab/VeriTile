import VeriTile.Triton.DSL.StaticPointerProfiles

/-! A one-argument Python range is the natural range from zero with step one.
Compound loads in identifier augmented assignments use the same typed-load
lowering as ordinary assignments. Original pasted source is unchanged. -/
namespace VeriTile.Triton.DSL.PastedRangeLoads
open Lean Elab Term Meta

private partial def containsLoad (raw : Syntax) : Bool :=
  let value : TSyntax `tritonExpr:=⟨raw⟩
  (match value with
    | `(tritonExpr| tl.load($_:tritonExpr $[, $_:tritonMemKwarg]*))=>Bool.true
    | `(tritonExpr| tl.load($_:tritonExpr, $_:tritonExpr $[, $_:tritonMemKwarg]*))=>Bool.true
    | _=>Bool.false) || raw.getArgs.any containsLoad

private partial def lower (raw : Syntax) : MacroM (Syntax × Bool) := do
  if raw.getKind==choiceKind then return ← lower raw[0]!
  let mut args:=#[]
  let mut changed:=Bool.false
  for arg in raw.getArgs do
    let (arg,used) ← lower arg
    args:=args.push arg
    changed:=changed || used
  let raw:=raw.setArgs args
  match raw with
  | `(tritonStmt| for $i:ident in range($stop:tritonExpr) { $body:tritonStmt* })=>
    return ((← `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr, 1) { $body* })).raw,Bool.true)
  | `(tritonStmt| $id:ident += $value:tritonExpr)=>
    if containsLoad value.raw then
      return ((← `(tritonStmt| $id:ident = $id:ident + $value:tritonExpr)).raw,Bool.true)
    return (raw,changed)
  | `(tritonStmt| $id:ident -= $value:tritonExpr)=>
    if containsLoad value.raw then
      return ((← `(tritonStmt| $id:ident = $id:ident - $value:tritonExpr)).raw,Bool.true)
    return (raw,changed)
  | `(tritonStmt| $id:ident *= $value:tritonExpr)=>
    if containsLoad value.raw then
      return ((← `(tritonStmt| $id:ident = $id:ident * $value:tritonExpr)).raw,Bool.true)
    return (raw,changed)
  | `(tritonStmt| $id:ident /= $value:tritonExpr)=>
    if containsLoad value.raw then
      return ((← `(tritonStmt| $id:ident = $id:ident / $value:tritonExpr)).raw,Bool.true)
    return (raw,changed)
  | _=>return (raw,changed)

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw:=prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok body=>pure body | .error message=>throwErrorAt raw "{message}"
  let (lowered,changed) ← liftMacroM (lower parsed)
  unless changed do throwUnsupportedSyntax
  let `(triton { $body:tritonStmt* }):=lowered | throwError "complete pasted source body required"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  elabTerm (prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.PastedRangeLoads

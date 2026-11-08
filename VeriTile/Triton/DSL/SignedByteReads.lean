import VeriTile.Triton.DSL.RegisterShapeBroadcast

/-! Native signed-byte inputs are represented by the unbounded integer memory
channel. Interpret each known byte read before it enters arithmetic. The int64
annotation below selects that mathematical channel; it is not a hardware load.
The emitted expression is signed8(load.int), with the original mask/padding.
Ambiguous pointer rebindings are rejected rather than guessed. -/
namespace VeriTile.Triton.DSL.SignedByteReads
open Lean Elab Term Meta
private partial def validate (regions : Inference.RegionDTypes) (pointers : Inference.PtrElems)
    (raw : Syntax) : MacroM Unit := do
  if let `(tritonStmt| $id:ident = $rhs:tritonExpr):=raw then
    if regions.any (·.1==id.getId.toString) then
      Macro.throwError "signed-byte storage ports cannot be rebound"
    if let some dtype:=pointers.lookup id.getId.toString then
      unless Inference.pointerElementDType? regions pointers rhs==some dtype do
        Macro.throwError "signed-byte read requires unambiguous pointer storage"
  if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*):=raw then
    if (#[a,b]++more).any (fun id=>pointers.any (·.1==id.getId.toString) || regions.any (·.1==id.getId.toString)) then
      Macro.throwError "signed-byte pointer tuple rebinding is unsupported"
  if raw.getNumArgs>1 && raw[0].isAtom && raw[1].isIdent &&
      ["for","tl.for"].contains raw[0].getAtomVal.trimAscii.toString &&
      pointers.any (·.1==raw[1].getId.toString) then
    Macro.throwError "signed-byte pointer cannot be shadowed by a loop counter"
  for child in raw.getArgs do validate regions pointers child
private partial def lower (regions : Inference.RegionDTypes) (pointers : Inference.PtrElems)
    (raw : Syntax) : MacroM (Syntax × Bool) := do
  if raw.getKind==choiceKind then
    let selected:=raw.getArgs.find? fun child=>match child with
      | `(tritonStmt| for $_:ident in range($($_:term), $($_:term), $($_:term)) { $_:tritonStmt* })=>Bool.true
      | _=>Bool.false
    return ← lower regions pointers (selected.getD raw[0]!)
  let mut changed:=Bool.false
  let mut args:=#[]
  for child in raw.getArgs do
    let (child,next) ← lower regions pointers child
    changed:=changed || next
    args:=args.push child
  let raw:=raw.setArgs args
  let wrap (p : TSyntax `tritonExpr) (mask : Option (TSyntax `tritonExpr))
      (kwargs : TSyntaxArray `tritonMemKwarg) : MacroM (Syntax × Bool) := do
    unless Inference.pointerElementDType? regions pointers p==some .int8 do return (raw,changed)
    if kwargs.any (fun kw=>kw.raw.getArgs.any (fun arg=>arg.isIdent && arg.getId.eraseMacroScopes==`dtype)) then
      return (raw,changed)
    let key:=mkIdent `dtype
    let dtype ← `(tritonMemKwarg| $key:ident = tl.int64)
    let kwargs:=kwargs.push dtype
    let loaded ← match mask with
      | some mask=>`(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))
      | none=>`(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))
    return ((← `(tritonExpr| tl.cast($loaded:tritonExpr, tl.int8))).raw,Bool.true)
  match raw with
  | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*))=>wrap p (some mask) kwargs
  | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*))=>wrap p none kwargs
  | _=>pure (raw,changed)
@[term_elab tritonBlock] def elaborateReads : TermElab := fun stx expected=>do
  let `(triton { $body:tritonStmt* }):=stx | throwUnsupportedSyntax
  unless body.any (fun stmt=>stmt.raw.getKind==``tritonCallSignature) do throwUnsupportedSyntax
  let regions:=Inference.collectRegionDTypes body.toList
  unless regions.any (·.2==DInfo.int8) do throwUnsupportedSyntax
  let pointers:=Inference.collectPtrElems regions body.toList
  let stx ← RegisterShapeBroadcast.prepare stx
  let (next,changed) ← liftMacroM do
    validate regions pointers stx
    lower regions pointers stx
  unless changed do throwUnsupportedSyntax
  elabTerm next expected
end VeriTile.Triton.DSL.SignedByteReads

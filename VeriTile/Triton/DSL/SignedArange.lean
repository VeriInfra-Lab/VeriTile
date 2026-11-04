import VeriTile.Triton.DSL.PastedRangeLoads

/-! Opt-in source typing for Triton's signed32 arange lanes. The source keeps
its ordinary tl.arange spelling. The existing integer carrier and conversion
implement that implicit type, including negative differences between lanes;
no subtraction may silently become natural saturating subtraction. -/
namespace VeriTile.Triton.DSL.SignedArange
open Lean Elab Term

private partial def arange? (value : TSyntax `tritonExpr) : Bool :=
  if value.raw.getKind==choiceKind then arange? ⟨value.raw[0]!⟩ else
  match value with
  | `(tritonExpr| ($inner:tritonExpr))=>arange? inner
  | `(tritonExpr| tl.arange($_:tritonExpr, $_:tritonExpr))
  | `(tritonExpr| tl.arange($_:tritonExpr))=>Bool.true
  | _=>Bool.false

private partial def lower (raw : Syntax) : MacroM (Syntax × Bool) := do
  if raw.getKind==choiceKind then return ← lower raw[0]!
  if let some (receiver,dtype):=methodCast? ⟨raw⟩ then
    let signed:=match dtype with | `(tritonDType| tl.int32)=>Bool.true | _=>Bool.false
    let direct:=arange? receiver
    if signed && direct then return (raw,Bool.false)
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.arange($a:tritonExpr, $b:tritonExpr)) =>
    return ((← `(tritonExpr| (tl.arange($a:tritonExpr, $b:tritonExpr)).to(tl.int32))).raw,Bool.true)
  | `(tritonExpr| tl.arange($b:tritonExpr)) =>
    return ((← `(tritonExpr| (tl.arange($b:tritonExpr)).to(tl.int32))).raw,Bool.true)
  | _ =>
    let mut args:=#[]
    let mut changed:=Bool.false
    for arg in raw.getArgs do
      let (arg,used) ← lower arg
      args:=args.push arg
      changed:=changed || used
    return (raw.setArgs args,changed)

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw:=prepared.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwErrorAt raw "{message}"
  let (parsed,changed) ← liftMacroM (lower parsed)
  unless changed do throwUnsupportedSyntax
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete source body"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (statements.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  elabTerm (prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.SignedArange

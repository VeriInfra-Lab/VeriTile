import VeriTile.Meta.StatementAudit
import VeriTile.Spec

/-! Bounded, conservative audit for a closed structural FP proof. Unlike the
instantiated-proof walker, this checks the entire project declaration graph,
including unused branches. It is intentionally unavailable when a proof might
use numerical constructors, opaque project code, nonstandard axioms or an
external FP certificate. Such proofs must use the ordinary assumption walker.
The trusted Lean/Mathlib boundary is the same as StatementAudit's. -/
namespace VeriTile.Meta.ClosedStructuralAudit
open Lean Elab Command

/-- Keep legacy execution relations visible too; a binder need not use the
new Spec wrapper to carry a numerical obligation. -/
private def fpPremiseType (name : Name) : Bool :=
  #[`VeriTile.Triton.FP.Equational.TermEq, `VeriTile.Triton.FP.Equational.OptionalEq,
    `VeriTile.Triton.FP.Structural.IO₁NumericalEquiv,
    `VeriTile.Triton.FP.Structural.ProgramIO.EquivalentOn,
    `VeriTile.Triton.FP.Guarded.Equivalent,
    `VeriTile.Triton.FP.GuardedRewrite.Equivalent,
    `VeriTile.Triton.FP.Guarded.Equivalent₁ₓ₂,
    `VeriTile.Triton.FP.Scheduled.Equivalent₁ₓ₂,
    `VeriTile.Triton.FP.Scheduled.Equivalent₃,
    `VeriTile.Triton.FP.Scheduled.Equivalent₁,
    `VeriTile.Triton.FP.Scheduled.ConditionalEquivalent₁,
    `VeriTile.Triton.FP.SoftmaxShift.LibdeviceExpSub,
    `VeriTile.Triton.FP.LogSumExpShift.IntrinsicLogMul,
    `VeriTile.Triton.FP.LogSumExpShift.LogLibdeviceExp,
    `VeriTile.Triton.FP.WelfordInduction.CountConversion,
    `VeriTile.Triton.FP.Structural.CellRelated,
    `VeriTile.Triton.FP.Structural.ValueRelated].contains name

private def forbidden (name : Name) : Bool :=
  #[``Spec.Derivation.atom, ``Spec.Derivation.strict,
    ``Spec.ProgramDerivation.numerical].contains name

/-- A binder can hide a certificate inside a project record, not just in its
headline type. Traverse constructor types as well as declaration definitions.
Inspect all branches conservatively and stop only at trusted dependencies. -/
private partial def premiseHasFP (work : List Name) (seen : NameSet := {}) : CoreM Bool := do
  match work with
  | [] => return false
  | name :: rest =>
    if seen.contains name then return ← premiseHasFP rest seen
    let seen := seen.insert name
    let env ← getEnv
    if isCoreConst env name then return ← premiseHasFP rest seen
    if (`VeriTile.Spec).isPrefixOf name || fpPremiseType name then return true
    let some info := env.find? name | return true
    let mut next ← specExprConsts env info.type
    if let some value := info.value? then next := next ++ (← specExprConsts env value)
    if let .inductInfo info := info then next := next ++ info.ctors.toArray
    premiseHasFP (next.toList ++ rest) seen

/-- This strict shortcut can print `none` only after rejecting every way a
closed project dependency could construct or conceal an atomic/strict step.
It does not infer that an empty declared rule table implies no assumptions. -/
elab "#print_fp_assumptions " id:ident " closed_structural" : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverload id
  let info ← liftCoreM <| getConstInfo name
  let (_, axioms) := ((CollectAxioms.collect name).run (← getEnv)).run {}
  unless axioms.axioms.all (#[`propext, `Classical.choice, `Quot.sound].contains ·) do
    throwError "closed structural audit requires the standard axiom base"
  liftTermElabM <| Meta.forallTelescope info.type fun params target => do
    unless target.isAppOf ``Spec.Equiv || target.isAppOf ``Spec.FloatingPoint do
      throwError "closed structural audit requires a public FP equivalence"
    let assumptions := target.getAppArgs[if target.isAppOf ``Spec.Equiv then 4 else 2]!
    unless (← Meta.whnf assumptions).isAppOf ``List.nil do
      throwError "closed structural audit requires an empty declared rule table"
    for param in params do
      let type ← Meta.inferType param
      if ← premiseHasFP (← specExprConsts (← getEnv) type).toList then
        throwError "closed structural audit rejects external FP proof/model premises"
  let deps ← liftCoreM <| specDependencies (← getEnv) [name]
  for dependency in deps.project do
    if forbidden dependency then
      throwError "closed structural audit found numerical evidence constructor {dependency}"
    match (← getEnv).find? dependency with
    | some (.opaqueInfo _) =>
      throwError "closed structural audit cannot inspect opaque project dependency {dependency}"
    | none => throwError "closed structural audit cannot inspect dependency {dependency}"
    | _ => pure ()
  logInfo m!"FP assumptions used by {id}:\n  none"

end VeriTile.Meta.ClosedStructuralAudit

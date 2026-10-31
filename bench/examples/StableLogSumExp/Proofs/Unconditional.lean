import bench.examples.StableLogSumExp.Proofs.UnconditionalModel

/-! A scalar countermodel for dropping the conditional log fallbacks. The
current kernel pair remains in Kernels.lean; no retired kernel is executable
here. These results refute an unguarded scalar rewrite, not the repaired pair. -/
namespace VeriTile.Bench.Examples.StableLogSumExpUnconditional
open VeriTile Triton FP.Structural FP.Guarded
open StableLogSumExpFPContract

/-- A scalar rewrite would have to hold in every model of its selected atoms
and at the admitted singleton input and intermediate operands. -/
def scalarGoal (assumptions : Spec.Assumptions FP.GuardedFragment) : Prop :=
  ∀ (M : Algebra Model.Number) (D : Domain Model.Number),
    Models assumptions M D →
    (requirements "x" 1 FP.Equational.seededSchedules).Holds M D Model.initial →
    StableLogSumExpFPExecution.stableValue
      (SoftmaxStableFPContract.engine M FP.Equational.seededSchedules)
      (fun _ : Fin 1 => Model.one) =
    StableLogSumExpFPExecution.directValue
      (SoftmaxStableFPContract.engine M FP.Equational.seededSchedules)
      (fun _ : Fin 1 => Model.one)

theorem unconditional_disproved (R : FP.Exponential.Rules) : ¬ scalarGoal R.assumptions := by
  intro h
  exact Model.scalar_outputs_different
    (h Model.algebra Model.domain (Model.exponential_models R) Model.domain_holds)

theorem unconditional_disproved_with_log_rules (R : FP.LogSumExp.Rules) :
    ¬ scalarGoal R.assumptions := by
  intro h
  exact Model.scalar_outputs_different
    (h Model.algebra Model.domain (Model.models R) Model.domain_holds)

#axiomsClean unconditional_disproved
#axiomsClean unconditional_disproved_with_log_rules
end VeriTile.Bench.Examples.StableLogSumExpUnconditional

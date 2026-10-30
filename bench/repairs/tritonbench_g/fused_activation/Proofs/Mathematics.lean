import bench.tritonbench_g.fused_activation.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FusedActivation.Mathematics
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FusedActivation
noncomputable section

private theorem some_lt_some (x y : ℝ) :
    @LT.lt (WithBot ℝ) WithBot.instLT (some x) (some y) ↔ x < y := WithBot.coe_lt_coe

theorem formula_eq (a : Bool) (x b i : ℝ) :
    fusedActivationSpec a x b i (1/2) = (let z := (1/2) * i + x + b; if a then Real.sigmoid z else max 0 z) := by
  cases a
  · by_cases h : (2 : ℝ)⁻¹ * i + x + b < 0
    · simp [fusedActivationSpec, fusedActivationInput,
        ComparableDType.gt, some_lt_some, h, max_eq_left (le_of_lt h)]
    · simp [fusedActivationSpec, fusedActivationInput,
        ComparableDType.gt, some_lt_some, h, max_eq_right (le_of_not_gt h)]
  · rfl

#axiomsClean formula_eq
end
end VeriTile.Bench.Repairs.FusedActivation.Mathematics

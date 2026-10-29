import bench.repairs.tritonbench_g.fused_activation.Proofs.Projection
import bench.tritonbench_g.fused_activation.Proofs.Real
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.FusedActivation.AccessSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.FusedActivation
noncomputable section

/-- Masked access bounds do not require separation between read-only regions.
Bias positivity is needed only when the output has an active element. -/
theorem fused_activation_kernel_access_safe (x b input : RegionName)
    (W N B : Nat) (a : Bool) (bounds : RegionBounds) (s : BlockState)
    (hw : 0 < N → 0 < W) (hx : N ≤ bounds x) (hb : W ≤ bounds b)
    (hi : N ≤ bounds input) :
    Kernel.TraceSafe bounds (kernel x b input W N B (1/2) a).toAlgKernel s := by
  rw [Projection.original]
  exact fused_add_mul_activation_kernel_traceSafe x b input W N B (1/2) a bounds s
    (fun _ hj => lt_of_lt_of_le hj hx)
    (fun _ hj => lt_of_lt_of_le (Nat.mod_lt _ (hw (Nat.zero_lt_of_lt hj))) hb)
    (fun _ hj => lt_of_lt_of_le hj hi)

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 24000 in
/-- Both source variants have the same actual active memory accesses under
all floating primitive interpretations. Numerical rewrite laws play no role. -/
theorem structural_traceSafe {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (x b input : RegionName) (W N B : Nat) (multiplier : ℝ) (a : Bool)
    (bounds : RegionBounds) (s : FP.Structural.State α)
    (hx : ∀ j : Fin B, s.pids 0 * B + j.val < N → s.pids 0 * B + j.val < bounds x)
    (hb : ∀ j : Fin B, s.pids 0 * B + j.val < N → (s.pids 0 * B + j.val) % W < bounds b)
    (hi : ∀ j : Fin B, s.pids 0 * B + j.val < N → s.pids 0 * B + j.val < bounds input) :
    FP.Structural.TraceSafe M bounds (kernel x b input W N B multiplier a) s ∧
      FP.Structural.TraceSafe M bounds (commutedKernel x b input W N B multiplier a) s := by
  open FP.Structural in
  cases a <;>
    simp (maxSteps := 1400000) [FP.Structural.TraceSafe, kernel, commutedKernel,
      ComputeKernel.surfaceBody, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeOp_unfold, Safety.safeCompute_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      store, Option.bind, numeric, bop, natLt, numericLt, integralMod,
      ofFloat, toFloat, Region.cast, State.setReg, ComputeDType.eraseDType]
  all_goals exact ⟨hx, hb, hi, hx⟩

#axiomsClean structural_traceSafe
#axiomsClean fused_activation_kernel_access_safe
end
end VeriTile.Bench.Repairs.FusedActivation.AccessSafety

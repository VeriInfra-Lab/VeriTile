import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelGrid
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.DequantizeMatmul.AccessSafety
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The wider scale mask is checked independently of the matrix mask. Both
actual sources have the same active load/store windows. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (version : Bool) (bounds : RegionName → Nat) (s : State α)
    (hb : ∀ i, i < p.K*p.N → i < bounds p.b)
    (hs : ∀ i, i < p.N → i < bounds p.scale)
    (ho : ∀ i, i < p.K*p.N → i < bounds p.output) :
    TraceSafe M bounds (kernel version p) s := by
  cases version
  all_goals
    simp (maxSteps := 1400000) [TraceSafe, kernel, original, candidate,
      TypedSource.kernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      step, evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind,
      numeric, bop, natLt, numericLt, numericEq, ofFloat, toFloat, Region.cast,
      State.setReg, ComputeDType.eraseDType, -Nat.cast_mul, -Nat.cast_add,
      ← Int.natCast_mul, ← Int.natCast_add, Int.ofNat_lt, Int.toNat_natCast,
      Nat.mul_comm p.BN]
    constructor
    · intro row col hr hc
      exact hb _ (by nlinarith)
    constructor
    · intro col hc
      exact hs _ hc
    · intro row col hr hc
      exact ho _ (by nlinarith)

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DequantizeMatmul.AccessSafety

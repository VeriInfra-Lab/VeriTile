import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Typed
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MaskedAddCuda.AccessSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000

/-- Both actual signed-mask source variants access only the declared bounded
windows. The veto narrows accesses after the initial integer mask load. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (scalar : ℝ)
    (bounds : RegionName → Nat) (s : State α)
    (hg : ∀ j : Fin p.block, s.pids 0 * p.block + j.val < p.n →
      s.pids 0 * p.block + j.val < bounds p.output)
    (hp : ∀ j : Fin p.block, s.pids 0 * p.block + j.val < p.n →
      s.pids 0 * p.block + j.val < bounds p.x)
    (hm : ∀ j : Fin p.block, s.pids 0 * p.block + j.val < p.n →
      s.pids 0 * p.block + j.val < bounds p.y) :
    TraceSafe M bounds (kernel storage p.output p.x p.y p.n scalar p.block) s := by
  cases storage
  all_goals
    simp (maxSteps := 1400000) [TraceSafe, kernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      step, evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind,
      numeric, bop, natLt, numericLt, numericEq, ofFloat, toFloat, Region.cast,
      State.setReg, ComputeDType.eraseDType]
    exact ⟨fun i hi => hm i hi, fun i hi _ => hp i hi,
      fun i hi _ => hg i hi⟩

/-- Both actual signed-mask source variants access only the declared bounded
windows. The veto narrows accesses after the initial integer mask load. -/
theorem commuted_safe {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (scalar : ℝ)
    (bounds : RegionName → Nat) (s : State α)
    (hg : ∀ j : Fin p.block, s.pids 0 * p.block + j.val < p.n →
      s.pids 0 * p.block + j.val < bounds p.output)
    (hp : ∀ j : Fin p.block, s.pids 0 * p.block + j.val < p.n →
      s.pids 0 * p.block + j.val < bounds p.x)
    (hm : ∀ j : Fin p.block, s.pids 0 * p.block + j.val < p.n →
      s.pids 0 * p.block + j.val < bounds p.y) :
    TraceSafe M bounds (commutedKernel storage p.output p.x p.y p.n scalar p.block) s := by
  cases storage
  all_goals
    simp (maxSteps := 1400000) [TraceSafe, commutedKernel, ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold, Safety.stepSafe_unfold, Safety.safeExpr_unfold,
      Safety.safeCompute_unfold, Safety.safeOp_unfold, Safety.safeAddress_unfold,
      Safety.safeMask_unfold, Safety.addressSafe_unfold, Safety.maskActive_unfold,
      step, evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind,
      numeric, bop, natLt, numericLt, numericEq, ofFloat, toFloat, Region.cast,
      State.setReg, ComputeDType.eraseDType]
    exact ⟨fun i hi => hm i hi, fun i hi _ => hp i hi,
      fun i hi _ => hg i hi⟩

#axiomsClean source_safe
#axiomsClean commuted_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MaskedAddCuda.AccessSafety

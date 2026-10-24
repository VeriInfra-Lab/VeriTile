import bench.repairs.tritonbench_g.mul_exponent_compensator.IO

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.SourceSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Actual evaluated masks bound both accesses, including empty vectors and
arbitrary initial registers. Inactive lanes require no backing allocation. -/
theorem source_safe (M : Algebra α) [Inhabited α] (p : Parameters)
    (bounds : RegionName → Nat)
    (input : ∀ i, i<p.n → i<bounds p.x)
    (output : ∀ i, i<p.n → i<bounds p.output) (s : State α) :
    TraceSafe M bounds (candidateKernel p.x p.output p.n p.block) s := by
  simp [TraceSafe,candidateKernel,sourceBody,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
    State.setReg,numericLt,natLt,bop,Region.cast,
    ComputeDType.eraseDType,Broadcast.leftIndex,Broadcast.rightIndex,
    input,output,numeric,← Int.natCast_mul,← Int.natCast_add,-Nat.cast_mul,-Nat.cast_add]
  refine ⟨fun i h=>input _ h,fun i h=>output _ h,?_⟩
  split <;> trivial

theorem schedule_safe (M : Algebra α) [Inhabited α] (src dst : RegionName)
    (n B : Nat) (bounds : RegionName → Nat)
    (input : ∀ i, i<n → i<bounds src)
    (output : ∀ i, i<n → i<bounds dst) (order : List Nat) (s : State α) :
    ScheduleSafe bounds (order.map (Interface.invocation M src dst n B)) s := by
  induction order generalizing s with
  | nil => trivial
  | cons pid rest ih =>
      exact ⟨source_safe M (Interface.parameters src dst n B) bounds input output _,
        fun t _ => ih t⟩

#axiomsClean source_safe
#axiomsClean schedule_safe
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.SourceSafety

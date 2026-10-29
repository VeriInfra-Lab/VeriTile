import bench.repairs.tritonbench_g.rmsnorm_fused.Proofs.Runtime
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.RmsnormFused.Runtime
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem source_trace_safe (A : Algebra α) (bounds : RegionName → Nat) (s : State α)
    (X Y W : RegionName) (N B : Nat) (eps : α) (hN : 0<N) (hNB : N≤B)
    (hx : ∀ j : Fin B, j.val<N → s.pids 0*N+j.val<bounds X)
    (hw : ∀ j : Fin B, j.val<N → j.val<bounds W)
    (hy : ∀ j : Fin B, j.val<N → s.pids 0*N+j.val<bounds Y) :
    TraceSafe A bounds (source X Y W N B) (bind s eps) := by
  have hB : B≠0 := by omega
  have hz : (0.0 : ℝ)=0 := by norm_num
  simp [TraceSafe,source,template,bind,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    Safety.rangeSafe_unfold,Safety.precisionContext,
    FP.Structural.exec,FP.Structural.run,FP.Structural.step,FP.Structural.range,evalExpr,evalComputeOp,evalOp_unfold,store,
    numeric,numericLt,numericLe,natLt,natLe,bop,ofFloat,toFloat,Region.cast,
    ComputeKernel.surfaceBody,ComputeDType.eraseDType,State.setReg,Option.bind,hN,hB,hz,Nat.not_lt.mpr hNB]
  simpa only [← Int.natCast_mul, Int.toNat_natCast] using ⟨hx, hw, hx, hy⟩
#axiomsClean source_trace_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFused.Runtime

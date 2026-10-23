import bench.repairs.tritonbench_g.rmsnorm_fused_llama.NativeIO
import bench.repairs.tritonbench_g.rmsnorm_fused_llama.Proofs.RuntimeGrid

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.NativeSafety
open VeriTile Triton FP.Structural FP.Structural.Safety
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem row (A : Algebra α) (candidate : Bool) (X Y W : RegionName) (N B : Nat)
    (eps : α) (s : State α) (hN : 0<N) (hNB : N≤B) (bounds : RegionName → Nat)
    (hx : ∀ k, k<N → s.pids 0*N+k<bounds X)
    (hw : ∀ k, k<N → k<bounds W)
    (hy : ∀ k, k<N → s.pids 0*N+k<bounds Y) :
    TraceSafe A bounds (NativeIO.source candidate X Y W N B)
      (s.setReg "eps_arg" .real [] (fun _ => eps)) := by
  have hB : B≠0 := by omega
  have hz : (0.0 : ℝ)=0 := by norm_num
  cases candidate
  all_goals
    simp (config := {maxSteps:=500000}) [NativeIO.source,runtimeCandidate,indexCommuted,
      runtimeKernel,VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.originalTyped,
      TraceSafe,traceSafeList_unfold,stepSafe_unfold,rangeSafe_unfold,
      safeExpr_unfold,safeCompute_unfold,safeOp_unfold,safeAddress_unfold,safeMask_unfold,
      addressSafe_unfold,maskActive_unfold,
      FP.Structural.exec,FP.Structural.run,FP.Structural.step,FP.Structural.range,
      evalExpr,evalComputeOp,evalOp_unfold,store,numeric,numericLt,numericLe,natLt,natLe,bop,
      ofFloat,toFloat,Region.cast,ComputeKernel.surfaceBody,ComputeDType.eraseDType,
      State.setReg,Option.bind,hN,hB,hz,Nat.not_lt.mpr hNB,
      Broadcast.leftIndex,Broadcast.rightIndex,hx,hw,hy,Nat.mul_comm]
  all_goals
    exact ⟨fun i hi => by simpa only [Nat.mul_comm] using hx i.val hi,
      fun i hi => hw i.val hi,
      fun i hi => by simpa only [Nat.mul_comm] using hx i.val hi,
      fun i hi => by simpa only [Nat.mul_comm] using hy i.val hi⟩

end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.NativeSafety

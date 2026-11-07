import bench.tritonbench_g.parallel_attention.Proofs.CallerCombined
import bench.tritonbench_g.parallel_attention.Proofs.CallerSafetyQueryWhole
import bench.tritonbench_g.parallel_attention.Proofs.CallerSafetyBackwardWhole

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller.SafetyProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 30000

theorem caller_prefix (M : Algebra α) (c : QueryConfig) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList M bounds (callerPrefix c) s := by
  simp [callerPrefix,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,run,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind]

theorem full_source (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α)
    (queryBounds : QueryBounds (config c s) bounds)
    (backwardBounds : BackwardBounds (config c s).toBackwardConfig bounds) :
    TraceSafe (withDot M D) bounds (kernel c) s := by
  obtain ⟨initial,hprefix,registers,_,_,_⟩:=prefix_run (withDot M D) c s
  obtain ⟨middle,hquery,_,nextRegisters⟩:=query_fp32 M D (config c s) initial registers
  have queryRun : run (withDot M D) (queryCode c) initial=some middle := by
    simpa only [queryCode,kernel_config] using hquery
  change Safety.traceSafeList _ _ _ _
  rw [source_split]
  simp only [List.append_assoc]
  rw [Safety.traceSafeList_append]
  refine ⟨caller_prefix (withDot M D) c bounds s,?_⟩
  intro u ran
  rw [hprefix] at ran
  cases Option.some.inj ran
  rw [Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · simpa only [queryCode,kernel_config] using query_source M D (config c s) bounds initial registers queryBounds
  · intro u ran
    rw [queryRun] at ran
    cases Option.some.inj ran
    simp only [List.singleton_append]
    apply Safety.traceSafeList_cons_of_step (t := middle)
    · simp [Safety.stepSafe_unfold]
    · simp [step]
    · simpa only [backwardCode,kernel_config] using
        backward_source M D (config c s) bounds middle nextRegisters backwardBounds

#axiomsClean caller_prefix
#axiomsClean full_source
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller.SafetyProof

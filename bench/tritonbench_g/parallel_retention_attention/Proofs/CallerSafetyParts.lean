import bench.tritonbench_g.parallel_retention_attention.Proofs.CallerParts
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Caller
open VeriTile.Triton FP.Structural
open PublicAccess (traceSafeList_of_steps)
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 5000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prologue_safe (M : Algebra α) (cap : RegionName → Nat)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    Safety.traceSafeList M cap (prologue c dq) s := by
  apply traceSafeList_of_steps
  simp [prologue,kernel,FPBackward.shellSource,backwardPastedKernel,ComputeKernel.surfaceBody]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  simp [Safety.safeOp_unfold]
  simp [step,evalExpr,evalComputeOp,Option.bind,evalOp_unfold,Safety.traceSafeList_unfold,
    Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

theorem query_arguments_safe (M : Algebra α) (cap : RegionName → Nat)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    Safety.traceSafeList M cap (queryArguments c dq) s := by
  apply traceSafeList_of_steps
  simp [queryArguments,kernel,FPBackward.shellSource,backwardPastedKernel,ComputeKernel.surfaceBody]
  simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

theorem key_arguments_safe (M : Algebra α) (cap : RegionName → Nat)
    (c : BackwardConfig) (dq : RegionName) (s : State α) :
    Safety.traceSafeList M cap (keyArguments c dq) s := by
  apply traceSafeList_of_steps
  simp [keyArguments,kernel,FPBackward.shellSource,backwardPastedKernel,ComputeKernel.surfaceBody]
  simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]

#axiomsClean prologue_safe
#axiomsClean query_arguments_safe
#axiomsClean key_arguments_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Caller

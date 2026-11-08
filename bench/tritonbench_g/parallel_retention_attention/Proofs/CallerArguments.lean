import bench.tritonbench_g.parallel_retention_attention.Proofs.CallerParts

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Caller
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)
noncomputable section
set_option maxHeartbeats 600000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem query_arguments_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) (r : Registers c s) :
    ∃ t,run M (queryArguments c dq) s=some t ∧ FPQuery.ArgumentReady c s t := by
  simp [queryArguments,kernel,FPBackward.shellSource,backwardPastedKernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,
    r.head,r.chunk,r.key,r.value,r.decay]
  constructor <;> simp [r.head,r.chunk,r.key,r.value,r.decay]

theorem key_arguments_run {α : Type} [Inhabited α] (M : Algebra α)
    (c : BackwardConfig) (dq : RegionName) (s : State α) (r : Registers c s) :
    ∃ t,run M (keyArguments c dq) s=some t ∧ FPStages.BackwardArgumentReady c s t := by
  simp [keyArguments,kernel,FPBackward.shellSource,backwardPastedKernel,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,
    r.head,r.chunk,r.key,r.value,r.decay]
  constructor <;> simp

#axiomsClean query_arguments_run
#axiomsClean key_arguments_run
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.Caller

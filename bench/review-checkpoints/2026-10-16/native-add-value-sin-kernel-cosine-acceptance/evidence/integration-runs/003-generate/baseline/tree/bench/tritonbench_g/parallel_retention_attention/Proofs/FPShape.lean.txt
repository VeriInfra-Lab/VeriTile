import bench.tritonbench_g.parallel_retention_attention.Proofs.FPBackwardWhole
namespace VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages
open VeriTile.Triton FP.Structural
open ParallelAttention.FPStages (ForwardConfig BackwardConfig)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

/-- The complete corrected forward transcription lowers successfully,
including its signed diagonal index difference. The old mathematical port
clipped this difference and represented -5 as 0-5, so this theorem does not
assert literal equality with that old source tree. -/
theorem forward_source_supported (c : ForwardConfig) :
    ∃ alg, (forwardKernel c).toAlgorithm? = Except.ok alg := by
  simp [forwardKernel, forwardSource, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, pure]

theorem backward_source_supported (c : BackwardConfig) :
    ∃ alg, (backwardKernel c).toAlgorithm? = Except.ok alg := by
  simp [backwardKernel, backwardSource, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Except.bind, bind, pure]

#axiomsClean forward_source_supported
#axiomsClean backward_source_supported
end
end VeriTile.Bench.TritonBenchG.ParallelRetentionAttention.FPStages

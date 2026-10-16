import bench.tritonbench_g.rope_embedding.Proofs.FPHalves
namespace VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem source_projection (backward second : Bool) (p : Parameters) :
    (source backward second p).eraseDType.surfaceBody =
      (legacySource backward second p).eraseDType.surfaceBody := by
  cases backward <;> cases second
  all_goals simp [source, legacySource, forwardFirst, forwardSecond, backwardFirst, backwardSecond,
    rope_embedding_forward_first_half, rope_embedding_forward_second_half,
    rope_embedding_backward_first_half, rope_embedding_backward_second_half,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
    VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, VeriTile.Triton.eraseDType]
  all_goals first
    | (conv_lhs => rw [Op.eraseDType.eq_def]; simp [Op.eraseDType, VeriTile.Triton.eraseDType])
    | (conv_rhs => rw [Op.eraseDType.eq_def]; simp [Op.eraseDType, VeriTile.Triton.eraseDType])
  all_goals simp only [Op.eraseDType.eq_def]
  all_goals rfl
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.RopeEmbedding.FPHalves

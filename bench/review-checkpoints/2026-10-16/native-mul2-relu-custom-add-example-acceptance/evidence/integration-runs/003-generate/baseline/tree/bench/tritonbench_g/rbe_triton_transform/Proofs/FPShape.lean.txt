import bench.tritonbench_g.rbe_triton_transform.Proofs.FPTyped
namespace VeriTile.Bench.TritonBenchG.RbeTritonTransform.FPTyped
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- All source statements, both barriers and both stores are retained; only
storage/compute types and the materialized implicit casts are erased. -/
theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  simp [source, legacySource, typed, rbe_triton_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
    VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [Op.eraseDType.eq_def]
  all_goals rfl
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.RbeTritonTransform.FPTyped

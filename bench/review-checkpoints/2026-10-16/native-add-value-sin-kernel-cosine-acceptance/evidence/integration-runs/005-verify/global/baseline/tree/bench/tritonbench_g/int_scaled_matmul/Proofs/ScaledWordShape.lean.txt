import bench.tritonbench_g.int_scaled_matmul.Proofs.ScaledWordSource
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord
open VeriTile.Triton
set_option maxHeartbeats 10000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
/-- Both constexpr arms are retained; the only loop annotations restore
signed8 storage and signed32 accumulation. -/
theorem body_materialization (p : Parameters) : body p=(originalBody p).map widths := by rfl

theorem prelude_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (prelude p)).eraseDType.toAlgKernel.body =
      smPreLoop p.input p.weight p.rows p.columns p.inputRowStride p.inputInnerStride
        p.weightInnerStride p.weightColumnStride p.blockRows p.blockColumns p.blockInner p.groupRows := by
  simp [prelude, original, typedScaled, smPreLoop, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem loop_projection (p : Parameters) :
    (ComputeKernel.mk [] [] (originalBody p)).eraseDType.toAlgKernel.body =
      smLoopBody p.inner p.inputInnerStride p.weightInnerStride p.blockRows p.blockColumns p.blockInner p.even := by
  simp [originalBody, original, typedScaled, smLoopBody, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

theorem store_index_projection (p : Parameters) :
    (ComputeKernel.mk [] [] ((epilogue p).take 6)).toAlgKernel.body =
      (smPostLoop p.output p.scale p.rows p.columns p.blockRows p.blockColumns).take 6 := by
  simp [epilogue, original, typedScaled, smPostLoop, ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  try rfl

#axiomsClean body_materialization
#axiomsClean prelude_projection
#axiomsClean loop_projection
#axiomsClean store_index_projection
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.ScaledWord

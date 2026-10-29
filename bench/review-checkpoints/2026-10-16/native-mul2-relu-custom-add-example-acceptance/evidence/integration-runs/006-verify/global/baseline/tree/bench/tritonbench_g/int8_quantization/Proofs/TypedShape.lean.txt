import bench.tritonbench_g.int8_quantization.Proofs.TypedScaled
namespace VeriTile.Bench.TritonBenchG.Int8Quantization.TypedScaled
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
/-- Same accepted supplied-scale stage, with fp32 storage/operation tags. -/
theorem source_shape (p : Parameters) :
    (kernel p).eraseDType.surfaceBody =
      (per_block_int8_store_slice p.x p.suppliedScale p.out p.scaleOut p.rows p.columns
        p.blockRows p.scaleStride p.preScale).eraseDType.surfaceBody := by
  simp [kernel, source, per_block_int8_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  repeat' constructor
  all_goals try simp [MemAccess.eraseDType, MaskOpt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl
#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.Int8Quantization.TypedScaled

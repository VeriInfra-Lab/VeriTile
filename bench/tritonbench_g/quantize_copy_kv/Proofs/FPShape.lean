import bench.tritonbench_g.quantize_copy_kv.Proofs.FPTypedSource

namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The original whole load/address/head-mask prefix, after float storage
annotation erasure. No column predicate or new source-value premise is added. -/
theorem load_prefix_projection (p : Parameters) :
    ((source p).eraseDType.toAlgKernel.body.take 5) =
      ((destindex_copy_quantize_kv_real_surface p.input p.destination p.output p.scales
        p.strideKbs p.strideKh p.strideKd p.strideObs p.strideOh p.strideOd
        p.strideSbs p.strideSh p.strideSd p.heads p.width p.height).toAlgKernel.body.take 5) := by
  simp only [source, destindex_copy_quantize_kv_real_surface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  rfl

/-- Both destination address calculations and both head-masked stores retain
the old source structure. The changed numerical producer is proved separately. -/
theorem store_tail_projection (p : Parameters) :
    ((source p).eraseDType.toAlgKernel.body.drop 8) =
      ((destindex_copy_quantize_kv_real_surface p.input p.destination p.output p.scales
        p.strideKbs p.strideKh p.strideKd p.strideObs p.strideOh p.strideOd
        p.strideSbs p.strideSh p.strideSd p.heads p.width p.height).eraseDType.toAlgKernel.body.drop 8) := by
  simp only [source, destindex_copy_quantize_kv_real_surface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  rfl

#axiomsClean load_prefix_projection
#axiomsClean store_tail_projection
end VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer

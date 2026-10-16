import bench.tritonbench_g.quantize_kv_transform.Proofs.TypedSource

namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Both supplied storage profiles retain the original complete load/address/
mask prefix. Subsequent abs and checked cast replace the old mathematical
projection, and the original fp16 scale store is preserved. -/
theorem load_prefix_projection (d : Storage) (p : Parameters) :
    ((source d p).eraseDType.toAlgKernel.body.take 5) =
      ((destindex_copy_quantize_kv_transform_real_surface p.input p.destination p.output p.scales
        p.strideKbs p.strideKh p.strideKd p.strideObs p.strideOh p.strideOd
        p.strideSbs p.strideSh 1 p.heads p.dimension p.width p.height).toAlgKernel.body.take 5) := by
  cases d
  all_goals simp only [source, source_fp16, source_fp32, destindex_copy_quantize_kv_transform_real_surface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  all_goals rfl

#axiomsClean load_prefix_projection
end VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform

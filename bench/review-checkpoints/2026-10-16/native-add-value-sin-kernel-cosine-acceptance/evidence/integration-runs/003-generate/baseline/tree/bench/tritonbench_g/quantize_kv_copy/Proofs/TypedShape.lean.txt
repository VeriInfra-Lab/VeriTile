import bench.tritonbench_g.quantize_kv_copy.Proofs.TypedQuantize

namespace VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The complete load/address/mask prefix is the accepted grouped source.
Later operations deliberately use the corrected floating abs and checked int8
cast, whose semantics differ from the historical mathematical projection. -/
theorem load_prefix_projection (p : Parameters) :
    ((source p).eraseDType.toAlgKernel.body.take 6) =
      ((destindex_copy_quantize_kv_group_real_surface p.input p.destination p.output p.scales
        p.strideKbs p.strideKh p.strideKg 1 p.strideObs p.strideOh p.strideOg 1
        p.strideSbs p.strideSh 1 p.groups p.height p.width).toAlgKernel.body.take 6) := by
  simp only [source, destindex_copy_quantize_kv_group_real_surface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    FloatDType.toTileDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, ComputeDType.eraseDType, Region.cast, Except.bind, bind]
  rfl

/-- The public typed source really reaches floating abs in fp32 context. -/
theorem source_abs_node (p : Parameters) :
    (source p).surfaceBody[6]? = some
      (.assign .real [p.height, p.width] "abs_data"
        (.compute (.alg .fp32 (.absFloat .real (.ref .real [p.height, p.width] "src_data"))))) := rfl

#axiomsClean load_prefix_projection
#axiomsClean source_abs_node
end VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize

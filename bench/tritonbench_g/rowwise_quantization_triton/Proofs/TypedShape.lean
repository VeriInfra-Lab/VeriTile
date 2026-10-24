import bench.tritonbench_g.rowwise_quantization_triton.Proofs.TypedScaled
namespace VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.TypedScaled
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
/-- Erasure checks the existing accepted stage's load/address/mask prefix. Floating
value correctness is proved separately, with the actual fp16 maximum store. -/
theorem source_load_prefix (p : Parameters) :
    (kernel p).eraseDType.surfaceBody.take 6 =
      (quantize_rowwise_scaled p.x p.out p.maxOut 0 p.columns p.width).eraseDType.surfaceBody.take 6 := by
  simp [kernel, source, quantize_rowwise_scaled, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
theorem source_abs_node (p : Parameters) :
    (kernel p).surfaceBody[6]? = some
      (.assign .real [p.width] "abs_x"
        (.compute (.alg .fp32 (.absFloat .real (.ref .real [p.width] "x"))))) := rfl

#axiomsClean source_load_prefix
#axiomsClean source_abs_node
end VeriTile.Bench.TritonBenchG.RowwiseQuantizationTriton.TypedScaled

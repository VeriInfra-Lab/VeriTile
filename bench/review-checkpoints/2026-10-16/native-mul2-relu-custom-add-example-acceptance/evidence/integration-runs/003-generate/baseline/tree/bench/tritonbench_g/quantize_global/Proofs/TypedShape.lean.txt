import bench.tritonbench_g.quantize_global.Proofs.TypedScaled
namespace VeriTile.Bench.TritonBenchG.QuantizeGlobal.TypedScaled
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
/-- Exact accepted scaled-stage correspondence at the source constant 127.
The llrint/final-int8 path is outside this pre-rounding contract. -/
theorem source_shape (X A O : RegionName) (N B : Nat) :
    (source X A O N B).eraseDType.surfaceBody =
      (quantize_global_scaled_store_slice X A O N B 127).eraseDType.surfaceBody := by
  simp [source, quantize_global_scaled_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure]
  repeat' constructor
  all_goals try simp [MemAccess.eraseDType, MaskOpt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals norm_num
#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.QuantizeGlobal.TypedScaled

import bench.tritonbench_g.quant_transpose_kernel.Proofs.TypedScaled
import bench.tritonbench_g.quant_transpose_kernel.Proofs.RealStoreSlice
local macro "true" : term => `(Bool.true)
local macro "false" : term => `(Bool.false)

namespace VeriTile.Bench.TritonBenchG.QuantTransposeKernel.TypedScaled
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem source_shape (A AbsmaxInv B : RegionName) (sam san sbn sbm M N BM BN : Nat) :
    (source A AbsmaxInv B sam san sbn sbm M N BM BN).eraseDType.surfaceBody =
      (quantize_global_transpose_scaled_store_slice A AbsmaxInv B sam san sbn sbm M N BM BN 127).eraseDType.surfaceBody := by
  simp [source, quantize_global_transpose_scaled_store_slice, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, show (127.0 : ℝ) = 127 by norm_num]
  all_goals repeat' constructor
  all_goals simp [Op.eraseDType.eq_def, Op.eraseDType_castFloat, VeriTile.Triton.eraseDType,
    show (127.0 : ℝ) = 127 by norm_num]
  all_goals rfl

#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.QuantTransposeKernel.TypedScaled

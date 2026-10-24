import bench.tritonbench_g.triton_argmax.Proofs.DimSource
import bench.tritonbench_g.triton_argmax.TritonArgmax
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.TritonArgmax.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 2500000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

/-- The first-stage source prefix is unchanged after dtype erasure. The final
store makes Python's implicit int64 pointer conversion explicit, fixing the
legacy port's Nat-valued store into its Int output region. -/
theorem stage1_prefix_shape (storage : Storage) (X Y : RegionName) (I : Region .int)
    (N B : Nat) (index64 : Bool) :
    (stage1 storage X Y I N B index64).eraseDType.surfaceBody.dropLast =
      (argmax_kernel_1 X Y I N B index64).eraseDType.surfaceBody.dropLast := by
  cases storage <;> cases index64 <;>
    simp [stage1, argmax_kernel_1, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
      Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
      Except.bind, bind, pure]
  all_goals repeat' constructor
  all_goals simp [Op.eraseDType.eq_def, Op.eraseDType_castFloat, VeriTile.Triton.eraseDType]
  all_goals rfl

theorem stage2_shape (storage : Storage) (X : RegionName) (I O : Region .int) (N B : Nat) :
    (stage2 storage X I O N B).eraseDType.surfaceBody =
      (argmax_kernel_2 X I O N B).eraseDType.surfaceBody := by
  cases storage <;>
    simp [stage2, argmax_kernel_2, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
      Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
      Except.bind, bind, pure]
  all_goals repeat' constructor
  all_goals simp [Op.eraseDType.eq_def, Op.eraseDType_castFloat, VeriTile.Triton.eraseDType]
  all_goals rfl

theorem dim_shape (storage : Storage) (X : RegionName) (O : Region .int)
    (M N K BM BN : Nat) (index64 : Bool) :
    (dim storage X O M N K BM BN index64).eraseDType.surfaceBody =
      (argmax_kernel X O M N K BM BN index64).eraseDType.surfaceBody := by
  cases storage <;> cases index64 <;>
    simp [dim, argmax_kernel, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
      Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
      Except.bind, bind, pure]
  all_goals repeat' constructor
  all_goals simp [Op.eraseDType.eq_def, Op.eraseDType_castFloat, VeriTile.Triton.eraseDType]
  all_goals rfl

#axiomsClean stage1_prefix_shape
#axiomsClean stage2_shape
#axiomsClean dim_shape
end VeriTile.Bench.TritonBenchG.TritonArgmax.TypedSource

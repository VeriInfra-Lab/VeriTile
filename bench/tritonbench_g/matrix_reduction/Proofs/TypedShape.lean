import bench.tritonbench_g.matrix_reduction.Proofs.TypedSource
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem source_shape (storage : Storage) (X Y : RegionName) (sm sn sy M N : Nat) :
    (kernel storage X Y sm sn sy M N).eraseDType.surfaceBody =
      (load_reduce_kernel X Y sm sn sy M N).eraseDType.surfaceBody := by
  cases storage <;>
    simp [kernel, source16, source32, load_reduce_kernel, ComputeKernel.eraseDType, source16, source32, ComputeKernel.toAlgKernel,
      ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
      Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
      ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
      Except.bind, bind, pure]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp [Op.eraseDType.eq_def, Op.eraseDType_castFloat, VeriTile.Triton.eraseDType]

#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.MatrixReduction.TypedSource

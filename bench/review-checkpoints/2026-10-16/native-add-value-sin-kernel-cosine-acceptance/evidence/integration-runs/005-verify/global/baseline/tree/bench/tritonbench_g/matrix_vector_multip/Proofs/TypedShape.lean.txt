import bench.tritonbench_g.matrix_vector_multip.Proofs.TypedContract
import bench.tritonbench_g.matrix_vector_multip.MatrixVectorMultip
namespace VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
@[simp] theorem erase_real_cast (e : Op .real shape) :
    (Op.castFloat .real .real e).eraseDType = e.eraseDType :=
  Op.eraseDType_castFloat .real .real e
/-- Full looping source correspondence, including the initial accumulator and its addition. Erasure checks only address/control shape, not numerical equivalence. -/
theorem source_shape (d : FloatDType) (p : Parameters) :
    (kernel d p).eraseDType.surfaceBody =
      (mv_kernel p.a p.b p.out p.rows p.columns p.strideRow p.strideColumn
        p.strideVector p.strideOut p.blockRows p.blockColumns).eraseDType.surfaceBody := by
  cases d
  all_goals try simp [kernel, source, mv_kernel, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp only [erase_real_cast, Op.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals try simp [MemAccess.eraseDType, MaskOpt.eraseDType, Op.eraseDType.eq_def, NumericDType.eraseDType, ComparableDType.eraseDType, VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl
#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.MatrixVectorMultip.Typed

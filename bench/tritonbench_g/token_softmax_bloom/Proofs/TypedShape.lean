import bench.tritonbench_g.token_softmax_bloom.Proofs.TypedContract
import bench.tritonbench_g.token_softmax_bloom.TokenSoftmaxBloom
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.Typed
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
@[simp] theorem erase_real_cast (e : Op .real shape) :
    (Op.castFloat .real .real e).eraseDType = e.eraseDType :=
  Op.eraseDType_castFloat .real .real e
/-- Complete source correspondence, including maximum, exp, sum, division and
store conversion. This erasure theorem asserts shape/control, not fp equality. -/
theorem source_shape (d : FloatDType) (p : Parameters) :
    (kernel d p).eraseDType.surfaceBody =
      (token_softmax_surface p.input p.start p.length p.output p.inputHeadStride
        p.inputTokenStride p.outputHeadStride p.outputTokenStride p.width).eraseDType.surfaceBody := by
  cases d
  all_goals simp [kernel, source, token_softmax_surface, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
    ComputeKernel.fromKernelBody, ComputeKernel.surfaceBody, ComputeDType.eraseDType,
    Except.bind, bind, pure, FloatDType.toTileDType, FloatDType.numericDType]
  all_goals repeat' constructor
  all_goals conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals simp [Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, VeriTile.Triton.eraseDType]
  all_goals simp only [FloatDType.toTileDType, MemAccess.eraseDType.eq_def,
    MaskOpt.eraseDType.eq_def, Op.eraseDType.eq_def, NumericDType.eraseDType,
    ComparableDType.eraseDType, VeriTile.Triton.eraseDType]
  all_goals rfl
#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.Typed

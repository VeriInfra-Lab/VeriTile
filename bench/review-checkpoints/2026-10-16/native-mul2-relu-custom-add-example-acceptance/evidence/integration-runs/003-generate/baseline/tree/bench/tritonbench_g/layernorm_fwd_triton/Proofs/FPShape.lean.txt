import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.layernorm_fwd_triton.Proofs.FPTypedSource
import bench.tritonbench_g.layernorm_fwd_triton.LayernormFwdTriton
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
@[simp] theorem erase_real_cast (e : Op .real shape) :
    (Op.castFloat .real .real e).eraseDType = e.eraseDType :=
  Op.eraseDType_castFloat .real .real e
/-- Source structure correspondence to the complete original accepted kernel.
This is a type-erasure bridge, not an assertion that erasing rounding preserves
floating values. The execution/value theorem uses the typed kernel itself. -/
theorem source_shape (p : Parameters) :
    (kernel p).eraseDType.surfaceBody =
      (layernorm_fwd_triton p.x p.weight p.y p.xSeqStride p.xHeadStride p.xDimStride
        p.ySeqStride p.yHeadStride p.yDimStride p.weightHeadStride p.weightDimStride
        p.columns p.block p.epsilon).eraseDType.surfaceBody := by
  simp [kernel, source, layernorm_fwd_triton, ComputeKernel.eraseDType,
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
  all_goals try simp [eq_mp_eq_cast, eq_mpr_eq_cast, cast_eq, id_eq]

#axiomsClean source_shape
end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages

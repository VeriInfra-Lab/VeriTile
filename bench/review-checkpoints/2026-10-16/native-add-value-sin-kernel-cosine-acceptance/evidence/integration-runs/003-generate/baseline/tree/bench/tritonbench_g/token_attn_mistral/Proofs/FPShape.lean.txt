import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.token_attn_mistral.Proofs.FPTypedSource
import bench.tritonbench_g.token_attn_mistral.TokenAttnMistral
namespace VeriTile.Bench.TritonBenchG.TokenAttnMistral.FPStages
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
      (token_attn_mistral_surface p.prob p.value p.output p.tokens p.request p.unusedStart p.length p.start p.attentionLength
        p.requestBatchStride p.requestTokenStride p.probHeadStride p.probTokenStride
        p.valueBatchStride p.valueHeadStride p.valueDimStride
        p.outputBatchStride p.outputHeadStride p.outputDimStride
        p.kvGroup p.window p.width p.block).eraseDType.surfaceBody := by
  simp [kernel, source, token_attn_mistral_surface, ComputeKernel.eraseDType,
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
end VeriTile.Bench.TritonBenchG.TokenAttnMistral.FPStages

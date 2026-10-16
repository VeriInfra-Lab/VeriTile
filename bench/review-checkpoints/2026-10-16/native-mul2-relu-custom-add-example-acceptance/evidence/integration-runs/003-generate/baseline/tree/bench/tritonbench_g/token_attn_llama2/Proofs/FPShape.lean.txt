import bench.tritonbench_g.token_attn_llama2.Proofs.FPTypedSource
import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.token_attn_llama2.TokenAttnLlama2
namespace VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) : ComputeKernel :=
  token_attn_llama2_surface p.q p.k p.scale p.locations p.starts p.lengths p.output p.maxLength
    p.locationBatchStride p.locationSequenceStride p.queryBatchStride p.queryHeadStride
    p.queryDimStride p.keyBatchStride p.keyHeadStride p.keyDimStride p.outputHeadStride
    p.outputSequenceStride p.groups p.dimensions p.tokens

/-- Full source control/address correspondence after erasing only floating
storage/compute annotations. Numerical correctness is proved separately. -/
theorem source_projection (p : Parameters) :
    (source p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  simp [source, legacySource, typed, token_attn_llama2_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
    VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType, VeriTile.Triton.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals rfl
#axiomsClean source_projection
end
end VeriTile.Bench.TritonBenchG.TokenAttnLlama2.FPValue

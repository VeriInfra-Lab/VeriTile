import bench.tritonbench_g.lora_expand_gemv.Proofs.FPTypedSource
import bench.tritonbench_g.lora_expand_gemv.LoraExpandGemv

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 28000
set_option linter.unusedSimpArgs false

def legacySource (p : Parameters) : ComputeKernel :=
  Full.bgmv_loop_surface p.input p.weight p.output p.indices p.inner p.length
    p.inputBatchStride p.inputInnerStride p.loraStride p.weightRowStride p.weightInnerStride
    p.outputBatchStride p.outputStride p.blockRows p.blockInner

/-- Whole-source control/address/mask bridge for all four input format pairs
and three output formats. Erasure is only a structural bridge; the concrete
arithmetic and casts are specified by the separate software execution theorem. -/
theorem source_projection (profile : InputProfile) (out : OutputFormat) (p : Parameters) :
    (source profile out p).eraseDType.surfaceBody = (legacySource p).eraseDType.surfaceBody := by
  cases profile <;> cases out
  all_goals simp [source, source_halfHalf, source_halfBfloat, source_bfloatHalf, source_bfloatBfloat,
    outputType, legacySource, Full.bgmv_loop_surface,
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
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue

import bench.tritonbench_g.lora_expand_gemv.Sources.Mathematical_BranchSource

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton
open FPValue (OutputFormat)
noncomputable section
set_option maxHeartbeats 40000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

/-- Inline only syntactically constant control branches. Dynamic guards, every
loop (with its original bounds), and every non-control statement are retained.
Recursing beneath control nodes accounts for the DSL's dtype-dependent
continuation duplication without deleting an executed statement. -/
def normalizeConst : List Stmt → List Stmt
  | [] => []
  | .ifThen (.constBool b) yes :: rest =>
      (if b then normalizeConst yes else []) ++ normalizeConst rest
  | .ifThenElse (.constBool b) yes no :: rest =>
      (if b then normalizeConst yes else normalizeConst no) ++ normalizeConst rest
  | .ifThen cond yes :: rest =>
      .ifThen cond (normalizeConst yes) :: normalizeConst rest
  | .ifThenElse cond yes no :: rest =>
      .ifThenElse cond (normalizeConst yes) (normalizeConst no) :: normalizeConst rest
  | .forLoop idx n body :: rest =>
      .forLoop idx n (normalizeConst body) :: normalizeConst rest
  | .forRange idx start stop stride body :: rest =>
      .forRange idx start stop stride (normalizeConst body) :: normalizeConst rest
  | .forRangeDyn idx start stop stride body :: rest =>
      .forRangeDyn idx start stop stride (normalizeConst body) :: normalizeConst rest
  | .forRangeInt idx start stop stride body :: rest =>
      .forRangeInt idx start stop stride (normalizeConst body) :: normalizeConst rest
  | stmt :: rest => stmt :: normalizeConst rest
termination_by code => sizeOf code
decreasing_by all_goals (simp_wf; try omega)

def fullLegacySource (p : Parameters) : ComputeKernel :=
  bgmv_expand_surface p.input p.weight p.output p.outputSize p.inner p.indices
    p.inputBatchStride p.inputInnerStride p.loraStride p.weightRowStride p.weightInnerStride
    p.outputBatchStride p.outputStride p.blockRows p.blockInner p.splits
    p.evenK p.addInputs p.castType

theorem sourceAlgorithmSupported (profile : InputProfile) (out : OutputFormat) (p : Parameters) :
    ∃ alg, (source profile out p).toAlgorithm? = Except.ok alg := by
  cases profile <;> cases out
  all_goals exact ⟨_, rfl⟩

theorem legacyAlgorithmSupported (p : Parameters) :
    ∃ alg, (fullLegacySource p).toAlgorithm? = Except.ok alg :=
  bgmv_expand_surface_toAlgorithm_supported _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

/-- Full source bridge, including the signed sentinel guard and all loop
statements. Only floating dtype erasure and constant-branch normalization are
used. This is a structural source relation, not a native arithmetic theorem. -/
theorem full_source_projection (profile : InputProfile) (out : OutputFormat) (p : Parameters) :
    normalizeConst (source profile out p).eraseDType.body =
      normalizeConst (fullLegacySource p).eraseDType.body := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases castType <;> cases addInputs <;> cases profile <;> cases out
  all_goals simp [source, fullLegacySource, bgmv_expand_surface,
    ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType,
    MemAccess.eraseDType, MaskOpt.eraseDType, ComputeKernel.fromKernelBody,
    ComputeKernel.surfaceBody, ComputeDType.eraseDType, Except.bind, bind, pure,
    VeriTile.Triton.eraseDType, FloatDType.numericDType, NumericDType.eraseDType,
    normalizeConst, show (0.0 : ℝ) = 0 by norm_num, FloatDType.toTileDType,
    ComparableDType.eraseDType]
  all_goals repeat' constructor
  all_goals try conv_lhs => rw [Op.eraseDType.eq_def]
  all_goals try simp [Op.eraseDType, VeriTile.Triton.eraseDType, normalizeConst,
    show (0.0 : ℝ) = 0 by norm_num, FloatDType.toTileDType,
    NumericDType.eraseDType, ComparableDType.eraseDType]
  all_goals try simp only [Op.eraseDType.eq_def]
  all_goals repeat' constructor
  all_goals rfl

#axiomsClean sourceAlgorithmSupported
#axiomsClean legacyAlgorithmSupported
#axiomsClean full_source_projection
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches

import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchReference
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 40000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue

def header (profile : InputProfile) (out : OutputFormat) (p : Parameters) :=
  (source profile out p).surfaceBody.take 3

def active (profile : InputProfile) (out : OutputFormat) (p : Parameters) : List ComputeStmt :=
  match (source profile out p).surfaceBody[3]? with
  | some (ComputeStmt.ifThen _ body) => body
  | _ => []

def castBranch (profile : InputProfile) (out : OutputFormat) (p : Parameters) : List ComputeStmt :=
  match (active profile out p).drop 4 with
  | ComputeStmt.ifThen _ yes :: after => (if p.castType then yes else []) ++ after
  | ComputeStmt.ifThenElse _ yes no :: after => (if p.castType then yes else no) ++ after
  | _ => []

def setup (profile : InputProfile) (out : OutputFormat) (p : Parameters) : List ComputeStmt :=
  (active profile out p).take 4 ++
    (castBranch profile out p).take ((castBranch profile out p).length-1)

def body (profile : InputProfile) (out : OutputFormat) (p : Parameters) : List ComputeStmt :=
  match (castBranch profile out p).getLast? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ b) => b
  | _ => []

def headerExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    {t : State Word // run (engine rest cfg plans) (header profile out p) s = some t} := by
  cases profile <;> cases out
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [header, source, ComputeKernel.surfaceBody, run, step, evalExpr,
      evalComputeOp, evalOp_unfold, Region.cast, Option.bind]
    rfl

def setupExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    {t : State Word // run (engine rest cfg plans) (setup profile out p)
      (headerExecution rest cfg plans profile out p s).val = some t} := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases evenK <;> cases castType <;> cases profile <;> cases out
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp [setup, active, castBranch, headerExecution, source, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
      bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
      engine, FPValue.backend, FPValue.formatOf]
    rfl

theorem setupInvariant (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    Invariant cfg profile p s (setupExecution rest cfg plans profile out p s).val := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases evenK <;> cases castType <;> cases profile <;> cases out
  all_goals constructor
  all_goals simp [setupExecution, dtype, effectiveFormat, inputFormat, weightFormat,
    aValue, aLoaded, aOffset, weightBase, selected, outputBase, length,
    ofFloat, toFloat, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
    Region.cast, show (0.0 : ℝ) = 0 by norm_num]
  all_goals try
    funext i
    simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex, Region.cast]
  all_goals try rfl

#axiomsClean headerExecution
#axiomsClean setupExecution
#axiomsClean setupInvariant
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches

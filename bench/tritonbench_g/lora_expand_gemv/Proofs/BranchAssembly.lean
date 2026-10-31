import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchExecution
import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchFrame
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
attribute [local simp] VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.length VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.promote VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.productFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.resultFormat
set_option maxRecDepth 40000
set_option maxHeartbeats 40000000
set_option linter.unusedSimpArgs false
local instance branchAssemblyWord : Inhabited Word := ⟨FPValue.Word.zero⟩

def guard (profile : InputProfile) (out : OutputFormat) (p : Parameters) : Op .bool [] :=
  match (source profile out p).surfaceBody[3]? with
  | some (ComputeStmt.ifThen cond _) => cond
  | _ => .constBool Bool.false

theorem sourceShape (profile : InputProfile) (out : OutputFormat) (p : Parameters) :
    (source profile out p).surfaceBody = header profile out p ++
      [.ifThen (guard profile out p) (active profile out p)] := by
  cases profile <;> cases out <;> rfl

def loopStmt (profile : InputProfile) (out : OutputFormat) (p : Parameters) : ComputeStmt :=
  .forRangeDyn "n" (.constNat 0) (.ref .nat [] "split_n_length") (.constNat p.blockRows)
    (body profile out p)

private theorem stepConstIf (M : Algebra Word) (b : Bool)
    (yes : List ComputeStmt) (s : State Word) :
    step M (.ifThen (.constBool b) yes) s =
      if b then run M yes s else some s := by
  conv_lhs =>
    rw [step]
    simp only [evalOp_unfold, bind, Option.bind, Option.pure_def]

private theorem stepConstIfElse (M : Algebra Word) (b : Bool)
    (yes no : List ComputeStmt) (s : State Word) :
    step M (.ifThenElse (.constBool b) yes no) s =
      if b then run M yes s else run M no s := by
  conv_lhs =>
    rw [step]
    simp only [evalOp_unfold, bind, Option.bind, Option.pure_def]

theorem runActive (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    run (engine rest cfg plans) (active profile out p) s =
      run (engine rest cfg plans) (setup profile out p ++ [loopStmt profile out p]) s := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases castType <;> cases profile <;> cases out
  all_goals simp [active, setup, castBranch, loopStmt, body, source, ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource,
    run_append, run, stepConstIf, stepConstIfElse, Option.bind_assoc]

theorem guardEval (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    evalOp (engine rest cfg plans) none (guard profile out p)
      (headerExecution rest cfg plans profile out p s).val =
      some (fun _ => decide (selected p s ≠ -1)) := by
  cases profile <;> cases out
  all_goals simp [guard, source, ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource, headerExecution,
    evalOp_unfold, numericEq, selected, ofFloat, toFloat, bop, Region.cast,
    Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals funext i
  all_goals simp [FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]

theorem headerMemory (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    (headerExecution rest cfg plans profile out p s).val.mem = s.mem := by
  cases profile <;> cases out <;> simp [headerExecution]

theorem setupMemory (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (s : State Word) :
    (setupExecution rest cfg plans profile out p s).val.mem = s.mem := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases evenK <;> cases castType <;> cases profile <;> cases out
  all_goals simp [setupExecution]

#axiomsClean sourceShape
#axiomsClean runActive
#axiomsClean guardEval
#axiomsClean headerMemory
#axiomsClean setupMemory
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Branches

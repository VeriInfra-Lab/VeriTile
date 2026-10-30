import bench.repairs.tritonbench_g.lora_expand_gemv.Proofs.BranchExecution
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
attribute [local simp] VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.length VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.promote VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.productFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.resultFormat
set_option maxRecDepth 40000
set_option maxHeartbeats 80000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FPValue.Word.zero⟩
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue

def bodyExecution (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial current : State Word)
    (n : Nat) (h : Invariant cfg profile p initial current) :
    {t : State Word // run (engine rest cfg plans) (body profile out p)
      (current.setReg "n" .nat [] (fun _ => n)) = some t} := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases castType <;> cases addInputs <;> cases profile <;> cases out
  all_goals
    have hav := h.av
    simp [dtype, effectiveFormat, inputFormat, weightFormat, FloatDType.toTileDType, ofFloat] at hav
    refine ⟨?_, ?_⟩
    swap
    simp [body, castBranch, active, source, ComputeKernel.surfaceBody,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, store, numeric,
      bop, natLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
      FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
      h.offk, h.offn, hav, h.bp, h.cp, h.size,
      engine, FPValue.backend, FPValue.formatOf, FPValue.typedSum]
    rfl

#axiomsClean bodyExecution
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Branches

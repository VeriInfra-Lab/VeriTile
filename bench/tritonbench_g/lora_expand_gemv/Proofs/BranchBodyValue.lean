import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchBody
namespace VeriTile.Bench.Repairs.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_scatter_regs)
noncomputable section
attribute [local simp] VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.source VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.length VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.inputFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.weightFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.effectiveFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.promote VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.productFormat VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches.resultFormat
set_option maxRecDepth 40000
set_option maxHeartbeats 80000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FPValue.Word.zero⟩
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue

theorem product_function (cfg : FP.Config) (p : Parameters)
    (initial : State Word) (mem : RegionName → Nat → Cell Word) (m : Nat) :
    product cfg .singleHalf p initial mem m = fun k =>
      FPValue.mulWord cfg (productFormat .singleHalf p)
        (FPValue.castWord cfg (effectiveFormat .singleHalf p) (productFormat .singleHalf p)
          (aValue cfg .singleHalf p initial k))
        (FPValue.castWord cfg (weightFormat .singleHalf) (productFormat .singleHalf p)
          (bValue .singleHalf p initial mem m k)) := rfl

theorem bodyMemory (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial current : State Word)
    (n : Nat) (h : Invariant cfg .singleHalf p initial current) :
    (bodyExecution rest cfg plans .singleHalf .single p initial current n h).val.mem =
      writeBlock cfg plans .singleHalf .single p initial current.mem n := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases castType <;> cases addInputs
  all_goals
    simp (config := { contextual := Bool.true }) [bodyExecution, writeBlock, masked_scatter_memory,
      FPValue.sumValues, outputCell, added, accumulated, product_function, bValue,
      productFormat, resultFormat, promote, effectiveFormat, dtype, inputFormat, weightFormat,
      outputOffset, outputBase, bOffset, weightBase, length,
      ofFloat, toFloat, FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      TileShape.insertAxisIndex, TileShape.dropInsertedIndex, TileShape.eraseAxis, TileShape.axisDim,
      show (0.0 : ℝ) = 0 by norm_num, FPValue.castWord_same, FPValue.castWord_zero]
  all_goals try rfl

theorem bodyInvariant (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial current : State Word)
    (n : Nat) (h : Invariant cfg .singleHalf p initial current) :
    Invariant cfg .singleHalf p initial
      (bodyExecution rest cfg plans .singleHalf .single p initial current n h).val := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases castType <;> cases addInputs
  all_goals constructor
  all_goals simp [bodyExecution, masked_scatter_regs, h.offk, h.offn, h.av, h.bp, h.cp, h.size]
  all_goals exact h.av

#axiomsClean bodyMemory
#axiomsClean bodyInvariant
end
end VeriTile.Bench.Repairs.LoraExpandGemv.Branches

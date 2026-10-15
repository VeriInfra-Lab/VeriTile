import bench.tritonbench_g.lora_expand_gemv.Proofs.BranchBody
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
open KcacheCopyTriton.TypedCopy (masked_scatter_memory masked_scatter_regs)
noncomputable section
set_option maxRecDepth 40000
set_option maxHeartbeats 80000000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
attribute [local irreducible] FPValue.castWord FPValue.mulWord FPValue.addWord FPValue.treeValue

theorem product_function (cfg : FP.Config) (profile : InputProfile) (p : Parameters)
    (initial : State Word) (mem : RegionName → Nat → Cell Word) (m : Nat) :
    product cfg profile p initial mem m = fun k =>
      FPValue.mulWord cfg (productFormat profile p)
        (FPValue.castWord cfg (effectiveFormat profile p) (productFormat profile p)
          (aValue cfg profile p initial k))
        (FPValue.castWord cfg (weightFormat profile) (productFormat profile p)
          (bValue profile p initial mem m k)) := rfl

theorem bodyMemory (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial current : State Word)
    (n : Nat) (h : Invariant cfg profile p initial current) :
    (bodyExecution rest cfg plans profile out p initial current n h).val.mem =
      writeBlock cfg plans profile out p initial current.mem n := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases castType <;> cases addInputs <;> cases profile <;> cases out
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
    (profile : InputProfile) (out : OutputFormat) (p : Parameters) (initial current : State Word)
    (n : Nat) (h : Invariant cfg profile p initial current) :
    Invariant cfg profile p initial
      (bodyExecution rest cfg plans profile out p initial current n h).val := by
  rcases p with ⟨input, weight, output, indices, inner, outputSize, splits,
    inputBatchStride, inputInnerStride, loraStride, weightRowStride, weightInnerStride,
    outputBatchStride, outputStride, blockRows, blockInner, evenK, addInputs, castType⟩
  cases castType <;> cases addInputs <;> cases profile <;> cases out
  all_goals constructor
  all_goals simp [bodyExecution, masked_scatter_regs, h.offk, h.offn, h.av, h.bp, h.cp, h.size]

#axiomsClean bodyMemory
#axiomsClean bodyInvariant
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches

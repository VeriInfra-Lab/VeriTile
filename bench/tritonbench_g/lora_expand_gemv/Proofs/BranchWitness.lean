import bench.tritonbench_g.lora_expand_gemv.BranchCorrect
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton FP.Structural
open FPValue (Word OutputFormat)
open FP.Equational (ReductionPlan)
noncomputable section
local instance branchWitnessWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- fp32 input is cast to half; ADD_INPUTS reads a weight/output alias. Two
rows in the first block collide at output address zero, then the second block
must add the first block's result. K is padded and the final row tile is partial. -/
def aliasParameters : Parameters where
  input := "inputs"
  weight := "weights_output"
  output := "weights_output"
  indices := "indices"
  inner := 1
  outputSize := 3
  splits := 1
  inputBatchStride := 2
  inputInnerStride := 1
  loraStride := 6
  weightRowStride := 2
  weightInnerStride := 1
  outputBatchStride := 6
  outputStride := 0
  blockRows := 2
  blockInner := 2
  evenK := Bool.false
  addInputs := Bool.true
  castType := Bool.true

def aliasInitial : State Word where
  mem := fun r _ =>
    if r = "indices" then .mk .int 0
    else if r = "inputs" then .mk .real (FP.Value.ofBits .fp32 0x3fc00000)
    else if r = "weights_output" then .mk .fp16 (FP.Value.ofBits .fp32 0x4000)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem alias_expected_value :
    expectedMemory .ieeeValue ReductionPlan.withSeed .singleHalf .half
      aliasParameters aliasInitial "weights_output" 0 =
      .mk .fp16 (FP.Value.ofBits .fp32 0x4800) := by
  change Cell.mk .fp16 _ = Cell.mk .fp16 _
  congr 1
  decide +kernel

theorem add_cast_alias_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (engine rest .ieeeValue ReductionPlan.withSeed)
        (source .singleHalf .half aliasParameters) aliasInitial = some final ∧
      final.mem "weights_output" 0 = .mk .fp16 (FP.Value.ofBits .fp32 0x4800) ∧
      final.mem "weights_output" 1 = aliasInitial.mem "weights_output" 1 ∧
      final.mem "inputs" = aliasInitial.mem "inputs" := by
  obtain ⟨final, he, hm, hf⟩ := correctness rest .ieeeValue ReductionPlan.withSeed
    .singleHalf .half aliasParameters aliasInitial (by decide)
  refine ⟨final, he, ?_, ?_, ?_⟩
  · rw [hm]
    exact alias_expected_value
  · apply hf
    right
    intro m
    simp [outputOffset, outputBase, aliasParameters, aliasInitial]
  · funext o
    exact hf "inputs" o (Or.inl (by decide))

def sentinelInitial : State Word :=
  { aliasInitial with mem := fun r o =>
      if r = "indices" then .mk .int (-1) else aliasInitial.mem r o }

theorem signed_sentinel_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (engine rest .ieeeValue ReductionPlan.withSeed)
        (source .singleHalf .half aliasParameters) sentinelInitial = some final ∧
      final.mem = sentinelInitial.mem := by
  obtain ⟨final, he, hm, _⟩ := correctness rest .ieeeValue ReductionPlan.withSeed
    .singleHalf .half aliasParameters sentinelInitial (by decide)
  refine ⟨final, he, ?_⟩
  rw [hm]
  simp [expectedMemory, selected, aliasParameters, sentinelInitial, aliasInitial, Region.cast]

#axiomsClean alias_expected_value
#axiomsClean add_cast_alias_execution
#axiomsClean signed_sentinel_execution
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches

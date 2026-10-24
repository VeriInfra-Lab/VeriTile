import bench.tritonbench_g.lora_expand_gemv.FPCorrect

namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Three output iterations, a partial final tile, a padded K lane, and real
input/output aliasing. Later blocks must use the input cached before any store. -/
def witnessParameters : Parameters where
  input := "io"
  weight := "weights"
  output := "io"
  indices := "indices"
  inner := 3
  length := 5
  inputBatchStride := 8
  inputInnerStride := 1
  loraStride := 40
  weightRowStride := 4
  weightInnerStride := 1
  outputBatchStride := 8
  outputStride := 1
  blockRows := 2
  blockInner := 4

def witnessState : State Word where
  mem := fun r o =>
    if r = "indices" then .mk .nat 0
    else if r = "io" then .mk .fp16 (FP.Value.ofBits .fp32
      (if o = 0 then 0x3c00 else if o = 1 then 0x4000 else if o = 2 then 0x4200 else 0x7e00))
    else if r = "weights" then .mk .fp16 (FP.Value.ofBits .fp32
      (if o % 4 = 1 then
        if o / 4 = 0 then 0x3c00 else if o / 4 = 1 then 0x4000
        else if o / 4 = 2 then 0x4200 else if o / 4 = 3 then 0x4400 else 0x4500
       else 0x3c00))
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

theorem witness_values :
    expected .ieeeValue ReductionPlan.withSeed .halfHalf .half witnessParameters witnessState 0 = FP.Value.ofBits .fp32 0x4600 ∧
    expected .ieeeValue ReductionPlan.withSeed .halfHalf .half witnessParameters witnessState 1 = FP.Value.ofBits .fp32 0x4800 ∧
    expected .ieeeValue ReductionPlan.withSeed .halfHalf .half witnessParameters witnessState 2 = FP.Value.ofBits .fp32 0x4900 ∧
    expected .ieeeValue ReductionPlan.withSeed .halfHalf .half witnessParameters witnessState 3 = FP.Value.ofBits .fp32 0x4a00 ∧
    expected .ieeeValue ReductionPlan.withSeed .halfHalf .half witnessParameters witnessState 4 = FP.Value.ofBits .fp32 0x4b00 := by
  decide +kernel

theorem multi_block_alias_execution (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (backend rest .ieeeValue ReductionPlan.withSeed)
        (source .halfHalf .half witnessParameters) witnessState = some final ∧
      final.mem "io" 0 = .mk .fp16 (FP.Value.ofBits .fp32 0x4600) ∧
      final.mem "io" 1 = .mk .fp16 (FP.Value.ofBits .fp32 0x4800) ∧
      final.mem "io" 2 = .mk .fp16 (FP.Value.ofBits .fp32 0x4900) ∧
      final.mem "io" 3 = .mk .fp16 (FP.Value.ofBits .fp32 0x4a00) ∧
      final.mem "io" 4 = .mk .fp16 (FP.Value.ofBits .fp32 0x4b00) ∧
      final.mem "io" 5 = witnessState.mem "io" 5 ∧
      final.mem "weights" = witnessState.mem "weights" := by
  obtain ⟨final,he,⟨hv,hf⟩,_⟩ := FPCorrect.lora_expand_full_machine_correctness
    rest .ieeeValue ReductionPlan.withSeed .halfHalf .half witnessParameters witnessState
    (by decide) (by decide) (by decide) (by decide)
  refine ⟨final,he,?_,?_,?_,?_,?_,?_,?_⟩
  · have h := hv ⟨0,by decide⟩
    simp only [outputCell, witness_values.1, outputType, ofFloat] at h
    simpa [outputOffset, outputBase, witnessParameters, witnessState] using h
  · have h := hv ⟨1,by decide⟩
    simp only [outputCell, witness_values.2.1, outputType, ofFloat] at h
    simpa [outputOffset, outputBase, witnessParameters, witnessState] using h
  · have h := hv ⟨2,by decide⟩
    simp only [outputCell, witness_values.2.2.1, outputType, ofFloat] at h
    simpa [outputOffset, outputBase, witnessParameters, witnessState] using h
  · have h := hv ⟨3,by decide⟩
    simp only [outputCell, witness_values.2.2.2.1, outputType, ofFloat] at h
    simpa [outputOffset, outputBase, witnessParameters, witnessState] using h
  · have h := hv ⟨4,by decide⟩
    simp only [outputCell, witness_values.2.2.2.2, outputType, ofFloat] at h
    simpa [outputOffset, outputBase, witnessParameters, witnessState] using h
  · apply hf
    right
    intro m
    have hm := m.isLt
    simp [outputOffset, outputBase, witnessParameters, witnessState]
    change m.val < 5 at hm
    omega
  · funext o
    exact hf "weights" o (Or.inl (by decide))

#axiomsClean witness_values
#axiomsClean multi_block_alias_execution
end
end VeriTile.Bench.TritonBenchG.LoraExpandGemv.FPValue

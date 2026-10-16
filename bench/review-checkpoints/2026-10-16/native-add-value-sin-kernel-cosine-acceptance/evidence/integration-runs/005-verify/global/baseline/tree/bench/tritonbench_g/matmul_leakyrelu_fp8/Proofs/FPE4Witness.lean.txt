import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.FPWitness
import bench.tritonbench_g.matmul_leakyrelu_fp8.Proofs.FPFusedWitness
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000

def e4WitnessState : State Word where
  mem := fun r o => if r=("A" : RegionName) ∨ r=("B" : RegionName) then
    .mk .f8e4 (FP.Value.ofBits .fp32 (if o=1 then 1 else 0x68))
    else .mk .fp16 (FP.Value.ofBits .fp32 0x4000)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- E4M3FN 64*64 produces 4096. The next nonzero product of two minimum
subnormals is 2^-18; adding it rounds back to 4096 in fp32. The final half
output is finite. This witness tests the concrete E4 decoding, not its name. -/
theorem e4_rounding_witness (enabled : Bool) :
    (TritonMatmul.FPValue.decode .e4m3 (FP.Value.ofBits .fp32 1)).toRat?=some (1/512 : Rat) ∧
    bitAccumulation (witnessParameters enabled) .e4m3 e4WitnessState (lane enabled)=FP.Value.ofBits .fp32 0x45800000 ∧
    bitOutput (witnessParameters enabled) .e4m3 e4WitnessState (lane enabled)=FP.Value.ofBits .fp32 0x6c00 := by
  cases enabled <;> decide +kernel

theorem e4_nonempty_execution (rest : Algebra Word) (enabled : Bool) :
    ∃ t, FP.Structural.exec (backend (software rest) sequentialDotAcc)
      (source (witnessParameters enabled) .e4m3) e4WitnessState=some t ∧
      t.mem "C" 0=Cell.mk .fp16 (FP.Value.ofBits .fp32 0x6c00) ∧
      t.mem "C" 1=e4WitnessState.mem "C" 1 := by
  obtain ⟨t,he,hv,hf⟩ := FPCorrect.software_correctness rest .e4m3 (witnessParameters enabled)
    e4WitnessState rfl (by change 2≤3; decide)
  have h := hv (lane enabled) (by cases enabled <;> decide)
  rw [(e4_rounding_witness enabled).2.2] at h
  refine ⟨t,he,?_,?_⟩
  · simpa [cOffset,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,e4WitnessState,BM,BN,lane,stored] using h
  · apply hf
    right
    intro i hi
    rcases i with ⟨i,j,u⟩
    simp [active,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,e4WitnessState,BM,BN] at hi
    have hi0 : i.val=0 := by omega
    have hj0 : j.val=0 := by omega
    simp [cOffset,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,e4WitnessState,BM,BN,hi0,hj0]

#axiomsClean e4_rounding_witness
#axiomsClean e4_nonempty_execution
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPFused
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000

def e4WitnessState : State Word where
  mem := fun r o => if r=("A" : RegionName) ∨ r=("B" : RegionName) then
    .mk .f8e4 (FP.Value.ofBits .fp32 (if o=1 then 1 else 0x68))
    else .mk .fp16 (FP.Value.ofBits .fp32 0x4000)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- E4M3FN 64*64 produces 4096. The next nonzero product of two minimum
subnormals is 2^-18; adding it rounds back to 4096 in fp32. The final half
output is finite. This witness tests the concrete E4 decoding, not its name. -/
theorem e4_rounding_witness (enabled : Bool) :
    (TritonMatmul.FPValue.decode .e4m3 (FP.Value.ofBits .fp32 1)).toRat?=some (1/512 : Rat) ∧
    bitAccumulation (witnessParameters enabled) .e4m3 e4WitnessState (lane enabled)=FP.Value.ofBits .fp32 0x45800000 ∧
    bitOutput (witnessParameters enabled) .e4m3 e4WitnessState (lane enabled)=FP.Value.ofBits .fp32 0x6c00 := by
  cases enabled <;> decide +kernel

theorem e4_nonempty_execution (rest : Algebra Word) (enabled : Bool) :
    ∃ t, FP.Structural.exec (backend (software rest) sequentialDotAcc)
      (source (witnessParameters enabled) .e4m3) e4WitnessState=some t ∧
      t.mem "C" 0=Cell.mk .fp16 (FP.Value.ofBits .fp32 0x6c00) ∧
      t.mem "C" 1=e4WitnessState.mem "C" 1 := by
  obtain ⟨t,he,hv,hf⟩ := FPCorrect.fused_software_correctness rest .e4m3 (witnessParameters enabled)
    e4WitnessState rfl (by change 2≤3; decide)
  have h := hv (lane enabled) (by cases enabled <;> decide)
  rw [(e4_rounding_witness enabled).2.2] at h
  refine ⟨t,he,?_,?_⟩
  · simpa [cOffset,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,e4WitnessState,BM,BN,lane,stored] using h
  · apply hf
    right
    intro i hi
    rcases i with ⟨i,j,u⟩
    simp [active,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,e4WitnessState,BM,BN] at hi
    have hi0 : i.val=0 := by omega
    have hj0 : j.val=0 := by omega
    simp [cOffset,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,e4WitnessState,BM,BN,hi0,hj0]

#axiomsClean e4_rounding_witness
#axiomsClean e4_nonempty_execution
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPFused

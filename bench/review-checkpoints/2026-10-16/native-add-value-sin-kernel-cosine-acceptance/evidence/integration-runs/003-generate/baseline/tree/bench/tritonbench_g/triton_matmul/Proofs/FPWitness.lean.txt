import bench.tritonbench_g.triton_matmul.FPCorrect
namespace VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters where
  A := "A"
  B := "B"
  C := "C"
  M := 1
  N := 1
  K := 2
  stride_am := 2
  stride_ak := 1
  stride_bk := 1
  stride_bn := 1
  stride_cm := 3
  stride_cn := 1
  BLOCK_SIZE_M := 2
  BLOCK_SIZE_N := 2
  BLOCK_SIZE_K := 1
  GROUP_SIZE_M := 2
  numKBlocks := 2

def oneCode : Storage → Word
  | .half => FP.Value.ofBits .fp32 0x3c00
  | .e4m3 => FP.Value.ofBits .fp32 56

def smallCode : Storage → Word
  | .half => FP.Value.ofBits .fp32 1
  | .e4m3 => FP.Value.ofBits .fp32 24

def witnessState (st : Storage) : State Word where
  mem := fun r o => stored st (if r=("A" : RegionName) ∧ o=1 then smallCode st else oneCode st)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : CIndex witnessParameters := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)

/-- The half input is a minimum subnormal; the second live-accumulator FMA
rounds 1+2^-24 back to 1. The fp8 accumulation is 17/16, then the output
conversion rounds its midpoint to the even 1 encoding. -/
theorem witness_rounding :
    bitAccumulation witnessParameters .half (witnessState .half) lane=FP.Value.ofBits .fp32 0x3f800000 ∧
    bitAccumulation witnessParameters .e4m3 (witnessState .e4m3) lane=FP.Value.ofBits .fp32 0x3f880000 ∧
    bitOutput witnessParameters .half (witnessState .half) lane=oneCode .half ∧
    bitOutput witnessParameters .e4m3 (witnessState .e4m3) lane=oneCode .e4m3 ∧
    (decode .half (smallCode .half)).toRat?=some (1/16777216 : Rat) ∧
    (decode .e4m3 (smallCode .e4m3)).toRat?=some (1/16 : Rat) := by
  decide +kernel

/-- Both original arms execute two K blocks. Invalid row/column lanes clamp
back to zero for loads and remain inactive for output; their store cells frame. -/
theorem nonempty_execution (rest : Algebra Word) (st : Storage) :
    ∃ t, FP.Structural.exec (backend (software rest) sequentialDotAcc) (source witnessParameters st)
      (witnessState st)=some t ∧
      t.mem "C" 0=stored st (oneCode st) ∧ t.mem "C" 1=(witnessState st).mem "C" 1 := by
  have hr : ∃ t, FP.Structural.exec (backend (software rest) sequentialDotAcc) (source witnessParameters st)
      (witnessState st)=some t ∧ Post witnessParameters st (witnessState st) t
        (bitOutput witnessParameters st (witnessState st)) := by
    cases st
    · exact FPCorrect.half_software_correctness rest witnessParameters (witnessState .half) rfl (by decide)
    · exact FPCorrect.e4m3_software_correctness rest witnessParameters (witnessState .e4m3) rfl (by decide)
  obtain ⟨t,he,hv,hf⟩ := hr
  have h := hv lane (by cases st <;> decide)
  have hvalue : bitOutput witnessParameters st (witnessState st) lane=oneCode st := by
    cases st
    · exact witness_rounding.2.2.1
    · exact witness_rounding.2.2.2.1
  rw [hvalue] at h
  refine ⟨t,he,?_,?_⟩
  · simpa [cOffset,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,witnessState,BM,BN,lane] using h
  · apply hf
    right
    intro i hi
    rcases i with ⟨i,j,u⟩
    simp [active,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,witnessState,BM,BN] at hi
    have hi0 : i.val=0 := by omega
    have hj0 : j.val=0 := by omega
    simp [cOffset,row,col,pidM,pidN,firstM,perGroup,blocksN,blocksM,groupSize,
      witnessParameters,witnessState,BM,BN,hi0,hj0]
/-- The witness lies in the exact old K/block and row-major layout domain. -/
theorem original_domain_witness :
    witnessParameters.K=BK witnessParameters*witnessParameters.numKBlocks ∧
    witnessParameters.stride_cn=1 ∧ BN witnessParameters≤witnessParameters.stride_cm := by decide

#axiomsClean witness_rounding
#axiomsClean nonempty_execution
#axiomsClean original_domain_witness
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.FPValue

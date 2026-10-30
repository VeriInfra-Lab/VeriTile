import bench.tritonbench_g.matmul_leakyrelu_fp8.FPCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def witnessParameters (enabled : Bool) : Parameters where
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
  ACTIVATION := enabled

def witnessState : State Word where
  mem := fun r o => .mk .fp16 (FP.Value.ofBits .fp32
    (if r=("A" : RegionName) then if o=1 then 0x8001 else 0xbc00 else 0x3c00))
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane (enabled : Bool) : CIndex (witnessParameters enabled) := (⟨0,by change 0<2; decide⟩,⟨0,by change 0<2; decide⟩,PUnit.unit)
def expected (enabled : Bool) : Word := FP.Value.ofBits .fp32 (if enabled then 0xa11f else 0xbc00)

/-- The second FMA loses the half subnormal -2^-24 at the -1 midpoint.
The enabled activation then uses the rounded binary32 0.01, rounds its product,
and narrows to binary16. Both branches execute nonempty multi-block input. -/
theorem witness_rounding (enabled : Bool) :
    bitAccumulation (witnessParameters enabled) .half witnessState (lane enabled)=FP.Value.ofBits .fp32 0xbf800000 ∧
    bitOutput (witnessParameters enabled) .half witnessState (lane enabled)=expected enabled := by
  cases enabled <;> decide +kernel

/-- Concrete activation edge policies: exact decimal 1/100 is rounded, ordered
comparison preserves negative zero, and NaN takes the scaled branch. -/
theorem activation_bit_witness :
    slope=FP.Value.ofBits .fp32 0x3c23d70a ∧
    slope.toRat?≠some (1/100 : Rat) ∧
    bitActivation Bool.true (FP.Value.zero .fp32 Bool.true)=FP.Value.zero .fp32 Bool.true ∧
    bitActivation Bool.true (FP.Value.qNaN .fp32)=FP.Value.qNaN .fp32 := by
  decide +kernel

theorem nonempty_execution (rest : Algebra Word) (enabled : Bool) :
    ∃ t, FP.Structural.exec (backend (software rest) sequentialDotAcc)
      (source (witnessParameters enabled) .half) witnessState=some t ∧
      t.mem "C" 0=stored .half (expected enabled) ∧ t.mem "C" 1=witnessState.mem "C" 1 := by
  obtain ⟨t,he,hv,hf⟩ := FPCorrect.software_correctness rest .half (witnessParameters enabled)
    witnessState rfl (by change 2≤3; decide)
  have h := hv (lane enabled) (by cases enabled <;> decide)
  rw [(witness_rounding enabled).2] at h
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

theorem original_domain_witness (enabled : Bool) :
    (witnessParameters enabled).K=BK (witnessParameters enabled)*2 ∧
    (witnessParameters enabled).stride_cn=1 ∧ BN (witnessParameters enabled)≤(witnessParameters enabled).stride_cm := by
  cases enabled <;> decide

#axiomsClean witness_rounding
#axiomsClean activation_bit_witness
#axiomsClean nonempty_execution
#axiomsClean original_domain_witness
end
end VeriTile.Bench.TritonBenchG.MatmulLeakyreluFp8.FPValue

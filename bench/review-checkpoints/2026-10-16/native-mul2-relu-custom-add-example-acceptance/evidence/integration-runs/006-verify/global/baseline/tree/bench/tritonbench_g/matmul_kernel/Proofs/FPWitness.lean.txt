import bench.tritonbench_g.matmul_kernel.FPCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue
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
  BLOCK_SIZE_M := 1
  BLOCK_SIZE_N := 1
  BLOCK_SIZE_K := 1
  numKBlocks := 2

def oneCode : Word := FP.Value.ofBits .fp32 0x3c00
def smallCode : Word := FP.Value.ofBits .fp32 1
def witnessState : State Word where
  mem := fun r o => stored .half (if r=("C" : RegionName) then FP.Value.ofBits .fp32 0x4000
    else if r=("A" : RegionName) ∧ o=1 then smallCode else oneCode)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : CIndex witnessParameters := (⟨0,by decide⟩,⟨0,by decide⟩,PUnit.unit)
/-- Two full K-blocks use live carry. The second fp32 fused multiply-add
rounds the exact value 1+2^-24 to 1; this is not identity rounding over Real. -/
theorem witness_rounding :
    bitAccumulation witnessParameters witnessState lane=FP.Value.ofBits .fp32 0x3f800000 ∧
    bitOutput witnessParameters witnessState lane=oneCode ∧
    (QuantizeKvTransform.MixedBits.widen smallCode).toRat?=some (1/16777216 : Rat) ∧
    (1+(1/16777216) : Rat) ≠ 1 := by
  decide +kernel

theorem nonempty_execution (rest : Algebra Word) :
    ∃ t, FP.Structural.exec (backend (software rest) sequentialDotAcc) (source witnessParameters .half)
      witnessState=some t ∧ t.mem "C" 0=stored .half oneCode ∧ t.mem "C" 1=witnessState.mem "C" 1 := by
  obtain ⟨t,he,hv,hf⟩ := FPCorrect.half_software_correctness rest witnessParameters witnessState (by decide)
  have h := hv lane
  rw [witness_rounding.2.1] at h
  refine ⟨t,he,?_,?_⟩
  · simpa [cOffset,row,col,pidM,pidN,witnessParameters,witnessState,BM,BN,lane] using h
  · apply hf
    right
    intro i
    rcases i with ⟨i,j,u⟩
    have hi : i.val=0 := by have := i.isLt; change i.val<1 at this; omega
    have hj : j.val=0 := by have := j.isLt; change j.val<1 at this; omega
    simp [cOffset,row,col,pidM,pidN,witnessParameters,witnessState,BM,BN,hi,hj]
/-- The two-block witness belongs to the original generalized theorem domain.
Python's fixed K=4096 host-trip binding is a separate specialization. -/
theorem original_domain_witness : BN witnessParameters≤4096 ∧ witnessParameters.numKBlocks>0 := by decide
#axiomsClean witness_rounding
#axiomsClean nonempty_execution
#axiomsClean original_domain_witness
end
end VeriTile.Bench.TritonBenchG.MatmulKernel.FPValue

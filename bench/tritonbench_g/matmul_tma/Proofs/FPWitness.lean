import bench.tritonbench_g.matmul_tma.Proofs.LegacyFP
namespace VeriTile.Bench.TritonBenchG.MatmulTma.FPValue
open VeriTile.Triton FP.Structural
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000

/-- Legal 16×16 tiles with transposed A strides, a padded output row stride,
and A/output aliasing. Both A and B are captured before any output is written. -/
def witnessParameters : Parameters :=
  ⟨"A", "B", "A", 16, 16, 16, 1, 16, 16, 1, 17, 1, 16, 16, 16⟩

def halfOperand : Word := QuantizeKvTransform.MixedBits.embed
  (MatmulDequantInt4.Binary16.Value.ofBits 0x3c01)

def witnessInitial : State Word where
  mem := fun r o => if r = "A" ∨ r = "B" then
      .mk .fp16 (if o < 16 then halfOperand else FP.Value.zero .fp32)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def witnessLane : CIndex witnessParameters := (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)

def expectedWord (half : Bool) : Word :=
  if half then FP.Value.ofBits .fp32 0x3c02 else FP.Value.ofBits .fp32 0x3f804008

theorem witness_bits (half : Bool) :
    bitOutput .ieeeValue sequentialDot witnessParameters half witnessInitial witnessLane =
      expectedWord half := by
  cases half <;> decide +kernel

theorem witness_layout : Function.Injective (cOffset witnessParameters) := by
  intro i j h
  rcases i with ⟨r,c,u⟩
  rcases j with ⟨r',c',u'⟩
  have hc := c.isLt
  have hc' := c'.isLt
  norm_num [witnessParameters] at hc hc'
  simp [cOffset, witnessParameters] at h
  apply Prod.ext
  · apply Fin.ext; change r.val = r'.val; omega
  · apply Prod.ext
    · apply Fin.ext; change c.val = c'.val; omega
    · exact Subsingleton.elim _ _

/-- Actual complete source execution, both output branches, a nonidentity
half-rounding boundary, and the untouched row-gap cell. This fixed regression
is supporting evidence for the arbitrary-parameter public theorems. -/
theorem witness_execution (rest : Algebra Word) (half : Bool) :
    ∃ t, FP.Structural.exec (backend (software rest .ieeeValue) sequentialDot)
      (source witnessParameters half) witnessInitial = some t ∧
      t.mem "A" 0 = .mk (outputDType half).toTileDType
        (ofFloat (outputDType half) (expectedWord half)) ∧
      t.mem "A" 16 = witnessInitial.mem "A" 16 ∧
      t.mem "B" 0 = .mk .fp16 halfOperand ∧
      (1025 / 1024 : Rat) * (1025 / 1024) ≠ 513 / 512 := by
  obtain ⟨t, he, hp⟩ := FPCorrect.matmul_tma_fp_software_correctness rest
    witnessParameters half witnessInitial witness_layout
  refine ⟨t, he, ?_, ?_, ?_, by decide +kernel⟩
  · have hv := hp.1 witnessLane
    rw [witness_bits] at hv
    simpa only [cOffset, witnessLane, witnessParameters,
      Nat.zero_mul, Nat.zero_add, Fin.val_zero] using hv
  · apply hp.2 "A" 16
    right
    intro i
    have hi := i.2.1.isLt
    norm_num [witnessParameters] at hi
    simp only [cOffset, witnessParameters, Nat.mul_one]
    omega
  · have hf := hp.2 "B" 0 (Or.inl (by decide))
    simpa [witnessInitial] using hf

#axiomsClean witness_bits
#axiomsClean witness_layout
#axiomsClean witness_execution
#specNonCircular outputValues avoiding [source, FP.Structural.exec, execution_memory]
#specNonCircular bitOutput avoiding [source, FP.Structural.exec, execution_memory]
end
end VeriTile.Bench.TritonBenchG.MatmulTma.FPValue

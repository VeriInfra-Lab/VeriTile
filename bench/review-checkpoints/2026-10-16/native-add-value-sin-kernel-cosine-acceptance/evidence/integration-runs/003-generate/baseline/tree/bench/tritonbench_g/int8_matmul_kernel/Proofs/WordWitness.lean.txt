import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordComplete
import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordBits
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 24000

abbrev witnessParameters : Parameters :=
  ⟨"input", "weight", "output", 15,16,128,128,1,16,1,17,1,16,16,32,1,1⟩
def witnessInitial : State Unit :=
  { mem := fun r _ => if r = "input" then .mk .int 255
      else if r = "weight" then .mk .nat 255 else .mk .int 91
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1
    undef := fun d _ _ => defaultValue d }

@[simp] theorem witness_offset (i : CIndex witnessParameters) :
    outputOffset witnessParameters witnessInitial i = 17*i.1.val+i.2.1.val := by
  simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
    witnessParameters, witnessInitial]
@[simp] theorem witness_active (i : CIndex witnessParameters) :
    active witnessParameters witnessInitial i ↔ i.1.val < 15 ∧ i.2.1.val < 16 := by
  simp [active, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
    witnessParameters, witnessInitial]

theorem witness_value : outerValues witnessParameters witnessInitial 4 (0,0,PUnit.unit) = -256 := by
  decide +kernel

theorem witness_injective : ∀ i j, active witnessParameters witnessInitial i →
    active witnessParameters witnessInitial j →
    outputOffset witnessParameters witnessInitial i = outputOffset witnessParameters witnessInitial j → i = j := by
  intro i j hi hj he
  simp only [witness_offset] at he
  have hi0 := i.1.isLt
  have hi1 := i.2.1.isLt
  have hj0 := j.1.isLt
  have hj1 := j.2.1.isLt
  have hrow : i.1.val = j.1.val := by omega
  have hcol : i.2.1.val = j.2.1.val := by omega
  exact Prod.ext (Fin.ext hrow) (Prod.ext (Fin.ext hcol) (Subsingleton.elim _ _))

/-- Full four-field/two-level source execution. The 16x16 tile and BK=32
exercise real signed narrowing; row 15 and the stride gap remain untouched.
This software witness is not a recorded GPU launch. -/
theorem nonempty_word_execution (M : Algebra Unit) :
    ∃ final, FP.Structural.exec (backend M) (source witnessParameters) witnessInitial = some final ∧
      final.mem "output" 0 = .mk .int (-256) ∧
      final.mem "output" 16 = .mk .int 91 ∧
      final.mem "output" 255 = .mk .int 91 := by
  obtain ⟨final, he, hv, hf⟩ := correctness M witnessParameters witnessInitial witness_injective
  refine ⟨final, he, ?_, ?_, ?_⟩
  · have v := hv (0,0,PUnit.unit) (by decide)
    simpa only [witness_offset, witness_value] using v
  · have outside : ∀ i, active witnessParameters witnessInitial i →
        16 ≠ outputOffset witnessParameters witnessInitial i := by
      intro i hi
      simp only [witness_offset]
      have hc := i.2.1.isLt
      change i.2.1.val < 16 at hc
      omega
    exact hf "output" 16 (Or.inr outside)
  · have outside : ∀ i, active witnessParameters witnessInitial i →
        255 ≠ outputOffset witnessParameters witnessInitial i := by
      intro i hi
      rw [witness_active] at hi
      simp only [witness_offset]
      omega
    exact hf "output" 255 (Or.inr outside)

#axiomsClean witness_value
#axiomsClean witness_injective
#axiomsClean nonempty_word_execution
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel

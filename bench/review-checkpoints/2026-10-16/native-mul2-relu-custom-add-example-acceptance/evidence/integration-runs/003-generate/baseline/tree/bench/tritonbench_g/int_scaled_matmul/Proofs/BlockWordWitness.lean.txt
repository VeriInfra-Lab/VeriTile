import bench.tritonbench_g.int_scaled_matmul.Proofs.BlockWordBits
namespace VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters :=
  ⟨"input", "weight", "output", 15, 31, 33, 33, 1, 31, 1, 33, 1, 16, 32, 32, 8⟩
def inputWord : BitVec 8 := 0xfe
def weightWord : BitVec 8 := 0x03

def witnessInitial : State Unit where
  mem := fun r _ => if r = ("weight" : RegionName) then .mk .int weightWord.toInt
    else if r = ("input" : RegionName) then .mk .int inputWord.toInt
    else .mk .int 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d r _ => match d with
    | .int => if r = ("input" : RegionName) then 5 else 7
    | d => defaultValue d

def lane : CIndex witnessParameters :=
  (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)

/-- K=33, BK=32: the second block has one normal product and 31 genuine
unspecified-byte products. Replacing padding by zero would incorrectly give -198. -/
theorem witness_result : outputValue witnessParameters witnessInitial lane = 887 := by decide +kernel

theorem witness_layout  :
    ∀ i j, active witnessParameters witnessInitial i → active witnessParameters witnessInitial j →
      outputOffset witnessParameters witnessInitial i = outputOffset witnessParameters witnessInitial j → i = j := by
  intro i j _ _ he
  rcases i with ⟨r,c,u⟩
  rcases j with ⟨r',c',u'⟩
  simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
    witnessParameters, witnessInitial] at he
  apply Prod.ext
  · apply Fin.ext
    change r.val = r'.val
    have hc := c.isLt
    have hc' := c'.isLt
    norm_num [witnessParameters] at hc hc'
    omega
  · apply Prod.ext
    · apply Fin.ext
      change c.val = c'.val
      have hc := c.isLt
      have hc' := c'.isLt
      norm_num [witnessParameters] at hc hc'
      omega
    · exact Subsingleton.elim _ _

theorem bit_success (rest : Algebra Unit) :
    ∃ final, FP.Structural.exec (backend rest)
      (source witnessParameters) witnessInitial = some final ∧
      final.mem "output" 0 = .mk .int 887 ∧
      final.mem "output" 31 = witnessInitial.mem "output" 31 ∧
      final.mem "output" 495 = witnessInitial.mem "output" 495 ∧
      final.mem "weight" 7 = .mk .int weightWord.toInt ∧
      (887 : Int) ≠ -198 := by
  obtain ⟨final, he, hp⟩ := correctness rest witnessParameters (by decide) witnessInitial witness_layout
  have hi : active witnessParameters witnessInitial lane := by decide
  refine ⟨final, he, ?_, ?_, ?_, ?_, by decide +kernel⟩
  · have ho : outputOffset witnessParameters witnessInitial lane = 0 := by decide
    simpa [ho, witness_result] using hp.1 lane hi
  · apply hp.2 "output" 31
    right
    intro i hi
    simp [active, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN, witnessParameters, witnessInitial] at hi
    have hc := i.2.1.isLt
    norm_num [witnessParameters] at hc
    simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial]
    omega
  · apply hp.2 "output" 495
    right
    intro i hi
    have hc := i.2.1.isLt
    norm_num [witnessParameters] at hc
    simp [active, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial] at hi
    simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial]
    omega
  · have hframe := hp.2 "weight" 7 (Or.inl (by change ("weight" : RegionName) ≠ "output"; decide))
    simpa [witnessInitial] using hframe

theorem witness_byte_inputs : ByteInputs witnessParameters witnessInitial := by
  constructor
  · intro k hk i hi
    exact ⟨inputWord, by simp [aPointer, witnessParameters, witnessInitial]⟩
  · intro k hk i hi
    exact ⟨weightWord, by simp [bPointer, witnessParameters, witnessInitial]⟩

theorem witness_bits : outputValue witnessParameters witnessInitial lane = (0x377 : BitVec 32).toInt :=
  witness_result

#axiomsClean witness_byte_inputs
#axiomsClean witness_bits
#axiomsClean witness_result
#axiomsClean witness_layout
#axiomsClean bit_success
end
end VeriTile.Bench.TritonBenchG.IntScaledMatmul.BlockWord

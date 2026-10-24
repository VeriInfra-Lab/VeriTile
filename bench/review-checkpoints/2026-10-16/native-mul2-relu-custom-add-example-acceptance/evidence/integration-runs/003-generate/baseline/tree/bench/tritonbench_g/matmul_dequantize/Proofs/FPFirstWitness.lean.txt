import bench.tritonbench_g.matmul_dequantize.Proofs.FPFirstSoftware
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPFirstSoftware
open VeriTile.Triton FP.Structural FPTypedFirst
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def witnessParameters (noGroups : Bool) : Parameters :=
  ⟨"input", "output", "scale", "weight", "zero", 15, 16, 32, 32, 1, 16, 1,
    17, 1, 16, 1, 2, 1, 16, 16, 16, 16, 1, noGroups⟩

def weightWord : BitVec 32 := 0x88888888
def zeroWord : BitVec 32 := 0xffffffff
def halfOne : Word := QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits 0x3c00)
def halfScale : Word := QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits 0x3555)
def halfResult (noGroups : Bool) : Word := QuantizeKvTransform.MixedBits.embed
  (MatmulDequantInt4.Binary16.Value.ofBits (if noGroups then 0xd4ab else 0xd8ab))

def witnessInitial : State Word where
  mem := fun r off => if r = ("weight" : RegionName) then .mk .int weightWord.toInt
    else if r = ("zero" : RegionName) then .mk .int zeroWord.toInt
    else if r = ("input" : RegionName) then .mk .fp16 halfOne
    else if r = ("scale" : RegionName) then .mk .fp16 (if off < 16 then halfScale else halfOne)
    else .mk .fp16 (QuantizeKvTransform.MixedBits.embed MatmulDequantInt4.Binary16.Value.zero)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane (mode : Bool) : CIndex (witnessParameters mode) := (⟨0, by change 0 < 16; decide⟩, ⟨0, by change 0 < 16; decide⟩, PUnit.unit)

/-- Both original branches, two K blocks and a partially masked M tile.
The grouped branch loads a different scale in its second K block. -/
theorem witness_result (mode : Bool) :
    machineOutput softwareDot (witnessParameters mode) witnessInitial (lane mode) = halfResult mode := by
  cases mode <;> decide +kernel

theorem witness_layout (mode : Bool) :
    ∀ i j, active (witnessParameters mode) witnessInitial i → active (witnessParameters mode) witnessInitial j →
      outputOffset (witnessParameters mode) witnessInitial i = outputOffset (witnessParameters mode) witnessInitial j → i = j := by
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

theorem bit_success (rest : Algebra Word) (mode : Bool) :
    ∃ final, FP.Structural.exec (FPTypedFirst.backend (scalar rest) softwareDot)
      (source (witnessParameters mode)) witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 (halfResult mode) ∧
      final.mem "output" 16 = witnessInitial.mem "output" 16 ∧
      final.mem "output" 255 = witnessInitial.mem "output" 255 ∧
      final.mem "weight" 7 = .mk .int weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -1195 / 512 := by
  obtain ⟨final, he, hp⟩ := correctness rest softwareDot (witnessParameters mode) witnessInitial (witness_layout mode)
  have hi : active (witnessParameters mode) witnessInitial (lane mode) := by cases mode <;> decide
  refine ⟨final, he, ?_, ?_, ?_, ?_, by decide +kernel⟩
  · have ho : outputOffset (witnessParameters mode) witnessInitial (lane mode) = 0 := by cases mode <;> decide
    simpa [ho, witness_result] using hp.1 (lane mode) hi
  · apply hp.2 "output" 16
    right
    intro i _
    have hc := i.2.1.isLt
    norm_num [witnessParameters] at hc
    simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial]
    omega
  · apply hp.2 "output" 255
    right
    intro i hi
    have hc := i.2.1.isLt
    norm_num [witnessParameters] at hc
    simp [active, inputRow, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial] at hi
    simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial]
    omega
  · have hframe := hp.2 "weight" 7 (Or.inl (by change ("weight" : RegionName) ≠ "output"; decide))
    simpa [witnessInitial] using hframe

#axiomsClean witness_result
#axiomsClean witness_layout
#axiomsClean bit_success
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPFirstSoftware

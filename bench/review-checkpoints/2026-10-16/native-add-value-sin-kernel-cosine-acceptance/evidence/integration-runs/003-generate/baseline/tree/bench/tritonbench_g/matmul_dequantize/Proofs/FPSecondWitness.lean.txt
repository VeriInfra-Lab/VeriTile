import bench.tritonbench_g.matmul_dequantize.Proofs.FPSecondSoftware
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPSecondSoftware
open VeriTile.Triton FP.Structural FPTypedSecond
open Int4Matmul.FPSoftware (Word scalar bounded_integer_cast dot)
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters :=
  ⟨"input", "output", "scale", "weight", "zero", 16, 16, 32, 32, 1, 16, 1,
    17, 1, 16, 1, 2, 1, 32, 16, 16, 16, 1, 2⟩

def weightWord : BitVec 32 := 0x88888888
def zeroWord : BitVec 32 := 0xffffffff
def halfOne : Word := QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits 0x3c00)
def halfScale : Word := QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits 0x3555)
def halfResult : Word := QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits 0xd4aa)

def witnessInitial : State Word where
  mem := fun r _ => if r = ("weight" : RegionName) then .mk .int weightWord.toInt
    else if r = ("zero" : RegionName) then .mk .int zeroWord.toInt
    else if r = ("input" : RegionName) then .mk .fp16 halfOne
    else if r = ("scale" : RegionName) then .mk .fp16 halfScale
    else .mk .fp16 (QuantizeKvTransform.MixedBits.embed MatmulDequantInt4.Binary16.Value.zero)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def lane : CIndex witnessParameters := (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)

/-- A legal 16×16 dot tile with two K blocks, negative packed int32 words,
a rounded half product and a final half store. -/
theorem witness_result : machineOutput softwareDot witnessParameters witnessInitial lane = halfResult := by
  decide +kernel

theorem witness_layout :
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

theorem bit_success (rest : Algebra Word) :
    ∃ final, FP.Structural.exec (FPTypedSecond.backend (scalar rest) softwareDot)
      (source witnessParameters) witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 halfResult ∧
      final.mem "output" 16 = witnessInitial.mem "output" 16 ∧
      final.mem "weight" 7 = .mk .int weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 := by
  obtain ⟨final, he, hp⟩ := correctness rest softwareDot witnessParameters witnessInitial witness_layout
  have hi : active witnessParameters witnessInitial lane := by decide
  refine ⟨final, he, ?_, ?_, ?_, by decide +kernel⟩
  · have ho : outputOffset witnessParameters witnessInitial lane = 0 := by decide
    simpa [ho, witness_result] using hp.1 lane hi
  · apply hp.2 "output" 16
    right
    intro i _
    have hc := i.2.1.isLt
    norm_num [witnessParameters] at hc
    simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial]
    omega
  · have hframe := hp.2 "weight" 7 (Or.inl (by decide))
    simpa [witnessInitial] using hframe

#axiomsClean witness_result
#axiomsClean witness_layout
#axiomsClean bit_success
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPSecondSoftware

import bench.tritonbench_g.int8_dequant_matmul.Proofs.FPSoftware
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPSoftware
open VeriTile.Triton FP.Structural FPTyped
open Int4Matmul.FPSoftware (Word)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

def witnessParameters (hasBias : Bool) : Parameters :=
  ⟨"input", "weight", "output", "bias", "rowScale", "columnScale", 15, 31, 64,
    64, 1, 32, 1, 33, 1, 16, 32, 32, 8, 2, hasBias⟩

def inputWord : BitVec 8 := 0x80
def weightWord : BitVec 8 := 0x7f
def divisor : Word := FP.Value.ofBits .fp32 0x38820610
def scaleOne : Word := FP.Value.ofBits .fp32 0x3f800000
def halfBias : Word := QuantizeKvTransform.MixedBits.embed (MatmulDequantInt4.Binary16.Value.ofBits 0x3555)
def halfResult (hasBias : Bool) : Word := QuantizeKvTransform.MixedBits.embed
  (MatmulDequantInt4.Binary16.Value.ofBits (if hasBias then 0xd403 else 0xd408))

def witnessInitial : State Word :=
  ({mem := fun r _ => if r = ("weight" : RegionName) then .mk .int weightWord.toInt
    else if r = ("input" : RegionName) then .mk .int inputWord.toInt
    else if r = ("rowScale" : RegionName) ∨ r = ("columnScale" : RegionName) then .mk .fp32 scaleOne
    else if r = ("bias" : RegionName) then .mk .fp16 halfBias
    else .mk .fp16 (QuantizeKvTransform.MixedBits.embed MatmulDequantInt4.Binary16.Value.zero)
    regs := fun _ _ _ => none
    pids := fun _ => 0
    numPids := fun _ => 1
    undef := fun d _ _ => defaultValue d} : State Word).setReg "divfactor" .fp32 [] (fun _ => divisor)

def lane (mode : Bool) : CIndex (witnessParameters mode) := (⟨0, by change 0 < 16; decide⟩, ⟨0, by change 0 < 32; decide⟩, PUnit.unit)

/-- Both bias branches, two full K blocks, signed byte inputs, and a partially
masked output tile. The final half result differs from exact rational arithmetic. -/
theorem witness_result (mode : Bool) :
    machineOutput (witnessParameters mode) witnessInitial divisor (lane mode) = halfResult mode := by
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
    ∃ final, FP.Structural.exec (FPTyped.backend (scalar rest))
      (source (witnessParameters mode)) witnessInitial = some final ∧
      final.mem "output" 0 = .mk .fp16 (halfResult mode) ∧
      final.mem "output" 31 = witnessInitial.mem "output" 31 ∧
      final.mem "output" 495 = witnessInitial.mem "output" 495 ∧
      final.mem "weight" 7 = .mk .int weightWord.toInt ∧
      ((-1040384 : Rat) / 16129) ≠ -129 / 2 := by
  obtain ⟨final, he, hp⟩ := correctness rest (witnessParameters mode) witnessInitial divisor (by simp [witnessInitial]) (witness_layout mode)
  have hi : active (witnessParameters mode) witnessInitial (lane mode) := by cases mode <;> decide
  refine ⟨final, he, ?_, ?_, ?_, ?_, by decide +kernel⟩
  · have ho : outputOffset (witnessParameters mode) witnessInitial (lane mode) = 0 := by cases mode <;> decide
    simpa [ho, witness_result] using hp.1 (lane mode) hi
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
    simp [active, inputRow, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial] at hi
    simp [outputOffset, rowBlock, columnBlock, firstRow, rowsInGroup, blocksM, blocksN,
      witnessParameters, witnessInitial]
    omega
  · have hframe := hp.2 "weight" 7 (Or.inl (by change ("weight" : RegionName) ≠ "output"; decide))
    simpa [witnessInitial] using hframe

theorem divisor_bits : FP.round .fp32 .ieeeValue (1 / 16129 : Rat) = divisor := by decide +kernel

theorem witness_bytes (mode : Bool) : ByteInputs (witnessParameters mode) witnessInitial := by
  constructor
  · intro k hk i
    exact ⟨inputWord, by simp [witnessInitial, witnessParameters]⟩
  · intro k hk i
    exact ⟨weightWord, by simp [witnessInitial, witnessParameters]⟩

#axiomsClean divisor_bits
#axiomsClean witness_bytes
#axiomsClean witness_result
#axiomsClean witness_layout
#axiomsClean bit_success
end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPSoftware

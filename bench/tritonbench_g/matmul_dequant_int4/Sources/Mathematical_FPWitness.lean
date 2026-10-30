import bench.tritonbench_g.matmul_dequant_int4.Sources.Mathematical_FPContract
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters :=
  ⟨"weight", "scale", "zero", "output", 8, 8, 8, 8, 1, 8, 1, 1, 1, 16, 1, 1, 1⟩
def weightWord : BitVec 32 := 0x80000000
def zeroWord : BitVec 32 := 0xf0000000
def scaleValue : Binary16.Value := Binary16.Value.ofBits 0x3555
def resultValue : Binary16.Value := Binary16.Value.ofBits 0xc0aa

def witnessInitial [Inhabited (Binary16.Value)] : State (Binary16.Value) where
  mem := fun r _ => if r = ("weight" : RegionName) then .mk .int weightWord.toInt
    else if r = ("zero" : RegionName) then .mk .int zeroWord.toInt
    else if r = ("scale" : RegionName) then .mk .fp16 scaleValue
    else .mk .fp16 (Binary16.Value.zero)
  regs := fun _ _ _ => none
  pids := fun _ => 7
  numPids := fun _ => 8
  undef := fun d _ _ => defaultValue d

def lane : Index witnessParameters := (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)

theorem witness_words : weightWord.toInt < 0 ∧ zeroWord.toInt < 0 ∧
    nibble weightWord.toInt 7 = 8 ∧ nibble zeroWord.toInt 7 = 15 := by decide +kernel

theorem witness_rounding :
    Binary16.mul .ieeeValue (Binary16.round .ieeeValue (-7)) scaleValue = resultValue ∧
    scaleValue.toRat? = some (1365 / 4096) ∧ resultValue.toRat? = some (-597 / 256) ∧
    (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 := by decide +kernel

theorem nonempty_software [Inhabited (Binary16.Value)] (rest : Algebra (Binary16.Value)) :
    ∃ final, FP.Structural.exec (software rest) (source witnessParameters) witnessInitial = some final ∧
      final.mem "output" 119 = .mk .fp16 resultValue ∧
      final.mem "output" 120 = witnessInitial.mem "output" 120 ∧
      final.mem "weight" 7 = .mk .int weightWord.toInt ∧
      (-7 * (1365 / 4096) : Rat) ≠ -597 / 256 := by
  obtain ⟨final, he, hp⟩ := software_correctness rest witnessParameters witnessInitial (by
    intro i j _ _ _
    exact Subsingleton.elim (α := Fin 1 × Fin 1 × PUnit) _ _)
  have hi : active witnessParameters witnessInitial lane := by
    simp [active, row, column, witnessParameters, witnessInitial, lane]
  refine ⟨final, he, ?_, ?_, ?_, witness_rounding.2.2.2⟩
  · have hv := hp.1 lane hi
    simpa [roundedValue, difference, weightOffset, zeroOffset, scaleOffset, outputOffset,
      row, column, witnessParameters, witnessInitial, lane,
      witness_words.2.2.1, witness_words.2.2.2, witness_rounding.1] using hv
  · apply hp.2 "output" 120
    right
    intro i _
    have hi := Subsingleton.elim (α := Fin 1 × Fin 1 × PUnit) i lane
    subst i
    simp [outputOffset, row, column, witnessParameters, witnessInitial, lane]
  · have hf := hp.2 "weight" 7 (Or.inl (by decide))
    simpa [witnessInitial] using hf

#axiomsClean witness_words
#axiomsClean witness_rounding
#axiomsClean nonempty_software
end
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPTyped

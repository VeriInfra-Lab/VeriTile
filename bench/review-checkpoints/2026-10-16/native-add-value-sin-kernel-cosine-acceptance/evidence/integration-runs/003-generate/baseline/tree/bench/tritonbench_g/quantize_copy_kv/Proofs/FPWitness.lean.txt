import bench.tritonbench_g.quantize_copy_kv.FPCorrect

namespace VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer
open VeriTile.Triton FP.Structural FP.IntegerConversion
open QuantizeKvTransform
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def witnessParameters : Parameters where
  input := "input"
  output := "quantized"
  scales := "scales"
  destination := "destination"
  strideKbs := 20
  strideKh := 4
  strideKd := 1
  strideObs := 100
  strideOh := 4
  strideOd := 1
  strideSbs := 10
  strideSh := 2
  strideSd := 99
  heads := 1
  height := 2
  width := 3

def witnessState : State Word where
  mem := fun r o =>
    if r = "destination" then .mk .nat 3
    else if r = "input" then .mk .fp16 (FP.Value.ofBits .fp32
      (if o = 20 then 0x3c00 else if o = 21 then 0xb800 else 0x3400))
    else if r = "scales" then .mk .fp16 (FP.Value.ofBits .fp32 0x1234)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 4
  undef := fun dtype _ _ => defaultValue dtype

theorem witness_quotients :
    toSigned 8 (quotient .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)) = some 127 ∧
    toSigned 8 (quotient .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit)) = some (-63) ∧
    toSigned 8 (quotient .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, ⟨2, by decide⟩, PUnit.unit)) = some 31 := by
  decide +kernel

theorem witness_scale_rounds :
    scale .ieeeValue witnessParameters witnessState ⟨0, by decide⟩ =
      FP.Value.ofBits .fp32 0x2008 ∧
    (MatmulDequantInt4.Binary16.Value.ofBits 0x2008).toRat? = some (129/16384) ∧
    (129/16384 : Rat) ≠ 1/127 := by
  constructor
  · decide +kernel
  constructor
  · decide +kernel
  · norm_num

theorem witness_inactive_conversion_undefined :
    toSigned 8 (quotient .ieeeValue witnessParameters witnessState
      (⟨1, by decide⟩, ⟨0, by decide⟩, PUnit.unit)) = none := by
  decide +kernel

theorem witness_value_injective : Function.Injective (outputOffset witnessParameters witnessState) := by
  intro a b he
  have haj := a.2.1.isLt
  have hbj := b.2.1.isLt
  simp [outputOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast] at he
  change a.2.1.val < 3 at haj
  change b.2.1.val < 3 at hbj
  have hg : a.1.val = b.1.val := by omega
  have hj : a.2.1.val = b.2.1.val := by omega
  exact Prod.ext (Fin.ext hg) (Prod.ext (Fin.ext hj) (Subsingleton.elim _ _))

theorem witness_scale_injective : Function.Injective (scaleOffset witnessParameters witnessState) := by
  intro a b he
  simp [scaleOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast] at he
  exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

/-- A nonempty three-column source tile and a padded head execute for every
undefined byte policy. The half scale really rounds 1/127; all three active
columns are stored, while both inactive destinations retain their old bits. -/
theorem bit_success_witness (rest : Algebra Word) (u : Word → BitVec 8) :
    ∃ t, FP.Structural.exec (FPProfile.backend rest .ieeeValue (bytePolicy u))
        (source witnessParameters) witnessState = some t ∧
      t.mem "quantized" 300 = .mk .int 127 ∧
      t.mem "quantized" 301 = .mk .int (-63) ∧
      t.mem "quantized" 302 = .mk .int 31 ∧
      t.mem "scales" 30 = .mk .fp16 (FP.Value.ofBits .fp32 0x2008) ∧
      t.mem "quantized" 304 = .mk .int 77 ∧
      t.mem "scales" 32 = .mk .fp16 (FP.Value.ofBits .fp32 0x1234) := by
  obtain ⟨t, he, hm, hv, hs, hf⟩ := quantize_copy_kv_machine_correctness rest .ieeeValue u
    witnessParameters witnessState (by decide) (by decide)
    witness_value_injective witness_scale_injective
  refine ⟨t, he, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have h := (hv (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit) (by decide)).1
    simp only [quantized, totalSigned, witness_quotients.1, Option.getD_some] at h
    simpa [outputOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast] using h
  · have h := (hv (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit) (by decide)).1
    simp only [quantized, totalSigned, witness_quotients.2.1, Option.getD_some] at h
    simpa [outputOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast] using h
  · have h := (hv (⟨0, by decide⟩, ⟨2, by decide⟩, PUnit.unit) (by decide)).1
    simp only [quantized, totalSigned, witness_quotients.2.2, Option.getD_some] at h
    simpa [outputOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast] using h
  · have h := hs (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit) (by decide)
    rw [witness_scale_rounds.1] at h
    simpa [scaleOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast] using h
  · rw [hf "quantized" 304 (Or.inr ?_) (Or.inl (by decide))]
    · rfl
    · intro i hi
      have hi' : i.1.val = 0 := by have := hi; simp [active, witnessParameters] at this; omega
      have hj : i.2.1.val < 3 := i.2.1.isLt
      simp [outputOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast, hi']
      omega
  · rw [hf "scales" 32 (Or.inl (by decide)) (Or.inr ?_)]
    · rfl
    · intro i hi
      have hi' : i.1.val = 0 := by simp [witnessParameters] at hi; omega
      simp [scaleOffset, destination, witnessParameters, witnessState, Cell.read, Region.cast, hi']

#axiomsClean witness_quotients
#axiomsClean witness_scale_rounds
#axiomsClean witness_inactive_conversion_undefined
#axiomsClean witness_value_injective
#axiomsClean witness_scale_injective
#axiomsClean bit_success_witness
end
end VeriTile.Bench.TritonBenchG.QuantizeCopyKv.TypedQuantizer

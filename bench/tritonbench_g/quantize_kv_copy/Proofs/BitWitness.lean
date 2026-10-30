import bench.tritonbench_g.quantize_kv_copy.Proofs.TypedQuantize

namespace VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def witnessParameters : Parameters where
  input := "input"
  output := "quantized"
  scales := "scales"
  destination := "destination"
  strideKbs := 20
  strideKh := 10
  strideKg := 2
  strideObs := 100
  strideOh := 10
  strideOg := 2
  strideSbs := 10
  strideSh := 2
  groups := 1
  height := 2
  width := 2

def witnessState : State Word where
  mem := fun r o =>
    if r = "destination" then .mk .nat 3
    else if r = "input" then .mk .real
      (if o = 30 then FP.Value.ofBits .fp32 0x42fe0000
       else if o = 31 then FP.Value.ofBits .fp32 0xc27e0000
       else FP.Value.ofBits .fp32 0xffa12345)
    else if r = "scales" then .mk .real (FP.Value.ofBits .fp32 0x12345678)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

private theorem witness_quotients :
    toSigned 8 (quotient .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)) = some 127 ∧
    toSigned 8 (quotient .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit)) = some (-63) := by
  decide +kernel

private theorem witness_scale :
    scale .ieeeValue witnessParameters witnessState ⟨0, by decide⟩ =
      FP.Value.ofBits .fp32 0x3f800000 := by
  decide +kernel

theorem witness_input_defined : InputDefined .ieeeValue witnessParameters witnessState := by
  intro i hi
  rcases i with ⟨g, d, unit⟩
  cases unit
  have hg : g = ⟨0, by decide⟩ := by
    apply Fin.ext
    change g.val = 0
    have := hi
    simp [active, witnessParameters] at this
    omega
  subst g
  fin_cases d
  · simp [witness_quotients.1]
  · simp [witness_quotients.2]

theorem witness_injective :
    ∀ a b, active witnessParameters a → active witnessParameters b →
      outputOffset witnessParameters witnessState a =
        outputOffset witnessParameters witnessState b → a = b := by
  intro a b ha hb he
  have ha0 : a.1.val = 0 := by have := ha; simp [active, witnessParameters] at this; omega
  have hb0 : b.1.val = 0 := by have := hb; simp [active, witnessParameters] at this; omega
  have he1 : a.2.1.val = b.2.1.val := by
    simpa [outputOffset, destination, witnessParameters, witnessState, Cell.read,
      Region.cast, ha0, hb0] using he
  exact Prod.ext (Fin.ext (ha0.trans hb0.symm))
    (Prod.ext (Fin.ext he1) (Subsingleton.elim _ _))

/-- Nonempty grouped execution and a padded zero-scale group. The original
inputs 127 and -63.5 produce scale 1, then int8 values 127 and -63; inactive
quantized/scale cells preserve their sentinels for every undefined policy. -/
theorem bit_success_witness (rest : Algebra Word) (u : Nat → Word → Int) :
    ∃ t, FP.Structural.exec (backend rest .ieeeValue u) (source witnessParameters)
          witnessState = some t ∧
      t.mem "quantized" 310 = .mk .int 127 ∧
      t.mem "quantized" 311 = .mk .int (-63) ∧
      t.mem "scales" 32 = .mk .real (FP.Value.ofBits .fp32 0x3f800000) ∧
      t.mem "quantized" 312 = .mk .int 77 ∧
      t.mem "scales" 33 = .mk .real (FP.Value.ofBits .fp32 0x12345678) := by
  let t := (execution rest .ieeeValue u witnessParameters witnessState (by decide)).val
  have hm := execution_memory rest .ieeeValue u witnessParameters witnessState (by decide)
    witness_input_defined
  refine ⟨t, (execution rest .ieeeValue u witnessParameters witnessState (by decide)).property,
    ?_, ?_, ?_, ?_, ?_⟩
  · change (execution rest .ieeeValue u witnessParameters witnessState _).val.mem _ _ = _
    rw [hm]
    have h := value_readback .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit) (by decide) (by decide) witness_injective
    simp only [expected, witness_quotients.1, Option.getD_some] at h
    simpa [outputOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read] using h
  · change (execution rest .ieeeValue u witnessParameters witnessState _).val.mem _ _ = _
    rw [hm]
    have h := value_readback .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit) (by decide) (by decide) witness_injective
    simp only [expected, witness_quotients.2, Option.getD_some] at h
    simpa [outputOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read] using h
  · change (execution rest .ieeeValue u witnessParameters witnessState _).val.mem _ _ = _
    rw [hm]
    have h := scale_readback .ieeeValue witnessParameters witnessState
      (⟨0, by decide⟩, PUnit.unit) (by decide)
    rw [witness_scale] at h
    simpa [scaleOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read] using h
  · change (execution rest .ieeeValue u witnessParameters witnessState _).val.mem _ _ = _
    rw [hm]
    have h := frame .ieeeValue witnessParameters witnessState "quantized" 312
      (Or.inr (by
        intro i hi
        simp [active, witnessParameters] at hi
        simp [outputOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read]
        have := i.2.1.isLt
        simp [witnessParameters] at this
        omega)) (Or.inl (by decide))
    simpa [witnessState] using h
  · change (execution rest .ieeeValue u witnessParameters witnessState _).val.mem _ _ = _
    rw [hm]
    have h := frame .ieeeValue witnessParameters witnessState "scales" 33
      (Or.inl (by decide)) (Or.inr (by
        intro i hi
        simp [witnessParameters] at hi
        simp [scaleOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read]
        omega))
    simpa [witnessState] using h

#axiomsClean witness_input_defined
#axiomsClean witness_injective
#axiomsClean bit_success_witness
end
end VeriTile.Bench.TritonBenchG.QuantizeKvCopy.TypedQuantize

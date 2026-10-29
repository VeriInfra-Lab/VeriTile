import bench.tritonbench_g.quantize_kv_transform.Proofs.Contract

namespace VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform
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
  strideKh := 4
  strideKd := 1
  strideObs := 100
  strideOh := 4
  strideOd := 1
  strideSbs := 10
  strideSh := 2
  heads := 1
  dimension := 2
  height := 2
  width := 3

def witnessState (d : Storage) : State Word where
  mem := fun r o =>
    if r = "destination" then .mk .nat 3
    else if r = "input" then
      match d with
      | .fp16 => .mk .fp16 (FP.Value.ofBits .fp32
          (if o = 20 then 0x57f0 else if o = 21 then 0xd3f0 else 0xfc01))
      | .fp32 => .mk .real (FP.Value.ofBits .fp32
          (if o = 20 then 0x42fe0000 else if o = 21 then 0xc27e0000 else 0xffa12345))
    else if r = "scales" then .mk .fp16 (FP.Value.ofBits .fp32 0x1234)
    else .mk .int 77
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 4
  undef := fun dtype _ _ => defaultValue dtype

private theorem witness_quotients (d : Storage) :
    toSigned 8 (quotient .ieeeValue d witnessParameters (witnessState d)
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit)) = some 127 ∧
    toSigned 8 (quotient .ieeeValue d witnessParameters (witnessState d)
      (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit)) = some (-63) := by
  cases d <;> decide +kernel

private theorem witness_scale (d : Storage) :
    scale .ieeeValue d witnessParameters (witnessState d) ⟨0, by decide⟩ =
      FP.Value.ofBits .fp32 0x3c00 := by
  cases d <;> decide +kernel

theorem witness_input_defined (d : Storage) :
    InputDefined .ieeeValue d witnessParameters (witnessState d) := by
  intro i hi
  rcases i with ⟨g, j, tail⟩
  cases tail
  have hg : g = ⟨0, by decide⟩ := by
    apply Fin.ext
    change g.val = 0
    have := hi.1
    simp [witnessParameters] at this
    omega
  subst g
  have hj : j.val = 0 ∨ j.val = 1 := by
    have := hi.2
    simp [witnessParameters] at this
    omega
  rcases hj with hj | hj
  · have he : j = ⟨0, by decide⟩ := Fin.ext hj
    subst j
    simp [(witness_quotients d).1]
  · have he : j = ⟨1, by decide⟩ := Fin.ext hj
    subst j
    simp [(witness_quotients d).2]

theorem witness_value_injective (d : Storage) :
    ∀ a b, active witnessParameters a → active witnessParameters b →
      outputOffset witnessParameters (witnessState d) a =
        outputOffset witnessParameters (witnessState d) b → a = b := by
  intro a b ha hb he
  have ha0 : a.1.val = 0 := by have := ha.1; simp [witnessParameters] at this; omega
  have hb0 : b.1.val = 0 := by have := hb.1; simp [witnessParameters] at this; omega
  have he1 : a.2.1.val = b.2.1.val := by
    simpa [outputOffset, destination, witnessParameters, witnessState, Cell.read,
      Region.cast, ha0, hb0] using he
  exact Prod.ext (Fin.ext (ha0.trans hb0.symm))
    (Prod.ext (Fin.ext he1) (Subsingleton.elim _ _))

theorem witness_scale_injective (d : Storage) :
    ∀ a b : TileIndex [witnessParameters.height, 1],
      a.1.val < witnessParameters.heads → b.1.val < witnessParameters.heads →
      scaleOffset witnessParameters (witnessState d) a =
        scaleOffset witnessParameters (witnessState d) b → a = b := by
  intro a b ha hb _
  have h0 : a.1.val = b.1.val := by simp [witnessParameters] at ha hb; omega
  exact Prod.ext (Fin.ext h0) (Subsingleton.elim _ _)

/-- Both supplied input formats execute a padded nonempty tile. The actual
half scale is 0x3c00; source 127 and -63.5 quantize to 127 and -63. Inactive
value and half-scale cells remain unchanged for every undefined policy. -/
theorem bit_success_witness (rest : Algebra Word) (u : Nat → Word → Int) (d : Storage) :
    ∃ t, FP.Structural.exec (MixedBits.backend rest .ieeeValue u) (source d witnessParameters)
          (witnessState d) = some t ∧
      t.mem "quantized" 300 = .mk .int 127 ∧
      t.mem "quantized" 301 = .mk .int (-63) ∧
      t.mem "scales" 30 = .mk .fp16 (FP.Value.ofBits .fp32 0x3c00) ∧
      t.mem "quantized" 302 = .mk .int 77 ∧
      t.mem "scales" 32 = .mk .fp16 (FP.Value.ofBits .fp32 0x1234) := by
  let t := (execution rest .ieeeValue u d witnessParameters (witnessState d) (by decide)).val
  have hm := execution_memory rest .ieeeValue u d witnessParameters (witnessState d) (by decide)
    (witness_input_defined d)
  refine ⟨t, (execution rest .ieeeValue u d witnessParameters (witnessState d) (by decide)).property,
    ?_, ?_, ?_, ?_, ?_⟩
  · change (execution rest .ieeeValue u d witnessParameters (witnessState d) _).val.mem _ _ = _
    rw [hm]
    have h := value_readback .ieeeValue d witnessParameters (witnessState d)
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit) (by decide) (by decide) (witness_value_injective d)
    simp only [expected, (witness_quotients d).1, Option.getD_some] at h
    simpa [outputOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read] using h
  · change (execution rest .ieeeValue u d witnessParameters (witnessState d) _).val.mem _ _ = _
    rw [hm]
    have h := value_readback .ieeeValue d witnessParameters (witnessState d)
      (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit) (by decide) (by decide) (witness_value_injective d)
    simp only [expected, (witness_quotients d).2, Option.getD_some] at h
    simpa [outputOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read] using h
  · change (execution rest .ieeeValue u d witnessParameters (witnessState d) _).val.mem _ _ = _
    rw [hm]
    have h := scale_readback .ieeeValue d witnessParameters (witnessState d)
      (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit) (by decide) (witness_scale_injective d)
    rw [witness_scale] at h
    simpa [scaleOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read] using h
  · change (execution rest .ieeeValue u d witnessParameters (witnessState d) _).val.mem _ _ = _
    rw [hm]
    have h := frame .ieeeValue d witnessParameters (witnessState d) "quantized" 302
      (Or.inr (by
        intro i hi
        simp [active, witnessParameters] at hi
        simp [outputOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read]
        omega)) (Or.inl (by decide))
    simpa [witnessState] using h
  · change (execution rest .ieeeValue u d witnessParameters (witnessState d) _).val.mem _ _ = _
    rw [hm]
    have h := frame .ieeeValue d witnessParameters (witnessState d) "scales" 32
      (Or.inl (by decide)) (Or.inr (by
        intro i hi
        simp [witnessParameters] at hi
        simp [scaleOffset, destination, witnessParameters, witnessState, Region.cast, Cell.read]
        omega))
    simpa [witnessState] using h

#axiomsClean witness_input_defined
#axiomsClean witness_value_injective
#axiomsClean witness_scale_injective
#axiomsClean bit_success_witness
end
end VeriTile.Bench.TritonBenchG.QuantizeKvTransform.TypedTransform

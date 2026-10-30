import bench.tritonbench_g.kv_cache_filling.Proofs.TypedUInt8

namespace VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedUInt8
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def witnessParameters : Parameters where
  input := "states"
  output := "cache"
  blockOffsets := "pages"
  sourceIndex := 2
  blockIndex := 1
  kvBlockIndex := 0
  strideSs := 10
  strideSh := 4
  strideSd := 1
  strideCn := 100
  strideCb := 10
  strideCh := 4
  strideCd := 1
  strideBoff := 1
  heads := 1
  dimension := 2
  height := 2
  width := 3

def witnessState : State Word where
  mem := fun r o =>
    if r = "pages" then .mk .nat 5
    else if r = "states" then .mk .real
      (if o = 20 then FP.Value.ofBits .fp32 0x3fe00000
       else if o = 21 then FP.Value.ofBits .fp32 0x437fe000
       else FP.Value.ofBits .fp32 0xffa12345)
    else .mk .nat 77
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private theorem truncation_one : toUnsigned 8 (FP.Value.ofBits .fp32 0x3fe00000) = some 1 := by
  decide +kernel
private theorem truncation_255 : toUnsigned 8 (FP.Value.ofBits .fp32 0x437fe000) = some 255 := by
  decide +kernel

theorem witness_input_defined : InputDefined witnessParameters witnessState := by
  intro i hi
  have h0 : i.1.val = 0 := by have := hi.1; simp [witnessParameters] at this; omega
  have h1 : i.2.1.val = 0 ∨ i.2.1.val = 1 := by
    have := hi.2; simp [witnessParameters] at this; omega
  rcases h1 with h1 | h1 <;>
    simp [inputValue, inputOffset, witnessParameters, witnessState, h0, h1,
      Cell.read, truncation_one, truncation_255]

theorem witness_injective :
    ∀ a b, active witnessParameters a → active witnessParameters b →
      outputOffset witnessParameters witnessState a =
        outputOffset witnessParameters witnessState b → a = b := by
  intro a b ha hb he
  have ha0 : a.1.val = 0 := by have := ha.1; simp [witnessParameters] at this; omega
  have hb0 : b.1.val = 0 := by have := hb.1; simp [witnessParameters] at this; omega
  have he1 : a.2.1.val = b.2.1.val := by
    simp [outputOffset, blockOffset, witnessParameters, witnessState, Cell.read,
      Region.cast, ha0, hb0] at he
    exact he
  exact Prod.ext (Fin.ext (ha0.trans hb0.symm))
    (Prod.ext (Fin.ext he1) (Subsingleton.elim _ _))

/-- Both original K/V stage shapes execute with nonempty stores. The source
fractions 1.75 and 255.875 truncate to 1 and 255. Padding contains a negative
signaling NaN in backing memory; its cache sentinel stays unchanged. The
undefined-conversion policy is arbitrary. -/
theorem uint8_success_witness (rest : Algebra Word) (u : Nat → Word → Nat)
    (isValue : Bool) :
    ∃ t, FP.Structural.exec (backend rest u) (source isValue witnessParameters)
          witnessState = some t ∧
      t.mem "cache" 510 = .mk .nat 1 ∧
      t.mem "cache" 511 = .mk .nat 255 ∧
      t.mem "cache" 512 = .mk .nat 77 := by
  let t := (execution rest u isValue witnessParameters witnessState).val
  refine ⟨t, (execution rest u isValue witnessParameters witnessState).property, ?_, ?_, ?_⟩
  · have h := readback rest u isValue witnessParameters witnessState witness_input_defined
        (⟨0, by decide⟩, ⟨0, by decide⟩, PUnit.unit) (by decide) witness_injective
    simpa [witnessParameters, witnessState, outputOffset, blockOffset, expected,
      inputValue, inputOffset, Region.cast, Cell.read, truncation_one] using h
  · have h := readback rest u isValue witnessParameters witnessState witness_input_defined
        (⟨0, by decide⟩, ⟨1, by decide⟩, PUnit.unit) (by decide) witness_injective
    simpa [witnessParameters, witnessState, outputOffset, blockOffset, expected,
      inputValue, inputOffset, Region.cast, Cell.read, truncation_255] using h
  · have h := frame rest u isValue witnessParameters witnessState witness_input_defined
        "cache" 512 (Or.inr (by
          intro i hi
          simp [active, witnessParameters] at hi
          simp [outputOffset, blockOffset, witnessParameters, witnessState, Region.cast, Cell.read]
          omega))
    simpa [witnessState] using h

#axiomsClean witness_input_defined
#axiomsClean witness_injective
#axiomsClean uint8_success_witness
end
end VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedUInt8

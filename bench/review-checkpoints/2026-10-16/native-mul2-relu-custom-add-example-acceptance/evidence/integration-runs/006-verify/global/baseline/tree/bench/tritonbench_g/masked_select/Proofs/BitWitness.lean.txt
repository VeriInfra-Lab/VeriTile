import bench.tritonbench_g.masked_select.Proofs.TypedCopy
import bench.tritonbench_g.kcache_copy_triton.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.TypedCopy (copyBackend bitsBackend)
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters where
  input := "input"
  output := "output"
  selection := "selection"
  prefixSum := "prefix"
  size := 7
  width := 4

/-- Program 1 sees offsets 4,5,6,7. Only 4 and 6 are selected/in-bounds.
The unselected lane 5 repeats prefix 2; lane 7 is masked out by the bound. -/
def witnessState : State (BitVec 32) where
  mem := fun r o =>
    if r = "selection" then .mk .bool (decide (o ≠ 5))
    else if r = "prefix" then .mk .nat (if o ≤ 5 then 2 else 3)
    else if r = "input" then .mk .real
      (if o = 4 then BitVec.ofNat 32 0x7f800001 else BitVec.ofNat 32 0x80000000)
    else .mk .real (BitVec.ofNat 32 0x12345678)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 2
  undef := fun d _ _ => defaultValue d

theorem witness_active (i : TileIndex [witnessParameters.width])
    (h : active witnessParameters witnessState i) : i.1.val = 0 ∨ i.1.val = 2 := by
  simp [active, inputOffset, witnessParameters, witnessState, Region.cast, Cell.read] at h
  omega

theorem witness_injective :
    ∀ a b, active witnessParameters witnessState a → active witnessParameters witnessState b →
      outputOffset witnessParameters witnessState a = outputOffset witnessParameters witnessState b → a = b := by
  intro a b ha hb hab
  have hval : a.1.val = b.1.val := by
    rcases witness_active a ha with ha0 | ha2 <;>
      rcases witness_active b hb with hb0 | hb2 <;>
      simp_all [outputOffset, inputOffset, witnessParameters, witnessState, Region.cast, Cell.read]
  exact Prod.ext (Fin.ext hval) (Subsingleton.elim _ _)

/-- A successful selection copies signaling NaN and negative zero payloads to
compacted output addresses, while both adjacent non-written cells are framed. -/
theorem bit_success_witness :
    ∃ t, FP.Structural.exec (copyBackend bitsBackend) (source .fp32 witnessParameters)
          witnessState = some t ∧
      t.mem "output" 1 = .mk .real (BitVec.ofNat 32 0x7f800001) ∧
      t.mem "output" 2 = .mk .real (BitVec.ofNat 32 0x80000000) ∧
      t.mem "output" 0 = .mk .real (BitVec.ofNat 32 0x12345678) ∧
      t.mem "output" 3 = .mk .real (BitVec.ofNat 32 0x12345678) := by
  let t := (execution bitsBackend .fp32 witnessParameters witnessState).val
  refine ⟨t, (execution bitsBackend .fp32 witnessParameters witnessState).property, ?_, ?_, ?_, ?_⟩
  · have h := readback bitsBackend .fp32 witnessParameters witnessState
        (⟨0, by decide⟩, PUnit.unit) (by decide) witness_injective
    simpa [witnessParameters, witnessState, outputOffset, inputOffset, Storage.tag, Region.cast, Cell.read] using h
  · have h := readback bitsBackend .fp32 witnessParameters witnessState
        (⟨2, by decide⟩, PUnit.unit) (by decide) witness_injective
    simpa [witnessParameters, witnessState, outputOffset, inputOffset, Storage.tag, Region.cast, Cell.read] using h
  · have h := frame bitsBackend .fp32 witnessParameters witnessState "output" 0
        (Or.inr (by
          intro i hi
          rcases witness_active i hi with h0 | h2 <;>
            simp [outputOffset, inputOffset, witnessParameters, witnessState, Region.cast, Cell.read, *]))
    simpa [witnessState] using h
  · have h := frame bitsBackend .fp32 witnessParameters witnessState "output" 3
        (Or.inr (by
          intro i hi
          rcases witness_active i hi with h0 | h2 <;>
            simp [outputOffset, inputOffset, witnessParameters, witnessState, Region.cast, Cell.read, *]))
    simpa [witnessState] using h

#axiomsClean bit_success_witness
end
end VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy

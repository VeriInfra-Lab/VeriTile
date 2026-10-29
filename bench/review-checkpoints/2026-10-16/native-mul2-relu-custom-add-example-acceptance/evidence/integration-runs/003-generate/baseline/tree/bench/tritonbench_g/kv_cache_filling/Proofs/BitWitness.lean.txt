import bench.tritonbench_g.kv_cache_filling.Proofs.TypedMetadata
import bench.tritonbench_g.kcache_copy_triton.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedMetadata
open VeriTile.Triton FP.Structural KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 2000000
set_option linter.unusedSimpArgs false

def witnessParameters : Parameters where
  input := "pre"
  output := "meta"
  blockOffsets := "offsets"
  blockIndex := 2
  kvBlockIndex := 1
  slot := 1
  strideMn := 128
  strideMb := 16
  strideMh := 2
  strideMd := 1
  strideBoff := 4
  heads := 2
  width := 4

def witnessState : State (BitVec 32) where
  mem := fun r o =>
    if r = "offsets" then .mk .nat 3
    else if r = "pre" then .mk .real
      (if o = 0 then BitVec.ofNat 32 0x3fc00000 else BitVec.ofNat 32 0x80000000)
    else .mk .real (BitVec.ofNat 32 0x12345678)
  regs := fun _ _ _ => none
  pids := fun _ => 1
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

theorem witness_injective : Function.Injective (outputOffset witnessParameters witnessState) := by
  intro a b h
  have hval : a.1.val = b.1.val := by
    simp [outputOffset, blockOffset, witnessParameters, witnessState, Region.cast, Cell.read] at h
    omega
  exact Prod.ext (Fin.ext hval) (Subsingleton.elim _ _)

/-- Nonempty metadata writeback, including preservation of an inactive head. -/
theorem bit_success_witness :
    ∃ t, FP.Structural.exec (copyBackend bitsBackend) (source .fp32 witnessParameters)
          witnessState = some t ∧
      t.mem "meta" 417 = .mk .real (BitVec.ofNat 32 0x3fc00000) ∧
      t.mem "meta" 419 = .mk .real (BitVec.ofNat 32 0x80000000) ∧
      t.mem "meta" 421 = .mk .real (BitVec.ofNat 32 0x12345678) := by
  let t := (execution bitsBackend .fp32 witnessParameters witnessState).val
  refine ⟨t, (execution bitsBackend .fp32 witnessParameters witnessState).property, ?_, ?_, ?_⟩
  · have h := readback bitsBackend .fp32 witnessParameters witnessState
        (⟨0, by decide⟩, PUnit.unit) (by decide) (fun _ _ _ _ hab => witness_injective hab)
    simpa [witnessParameters, witnessState, outputOffset, blockOffset, Storage.tag, Region.cast, Cell.read] using h
  · have h := readback bitsBackend .fp32 witnessParameters witnessState
        (⟨1, by decide⟩, PUnit.unit) (by decide) (fun _ _ _ _ hab => witness_injective hab)
    simpa [witnessParameters, witnessState, outputOffset, blockOffset, Storage.tag, Region.cast, Cell.read] using h
  · have h := frame bitsBackend .fp32 witnessParameters witnessState "meta" 421
        (Or.inr (by
          intro i hi
          simp [active, witnessParameters] at hi
          simp [outputOffset, blockOffset, witnessParameters, witnessState, Region.cast, Cell.read]
          omega))
    simpa [witnessState] using h

#axiomsClean bit_success_witness
end
end VeriTile.Bench.TritonBenchG.KvCacheFilling.TypedMetadata

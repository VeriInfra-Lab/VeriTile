import bench.tritonbench_g.kv_cache_copy.Proofs.TypedCopy
import bench.tritonbench_g.kcache_copy_triton.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.KvCacheCopy.TypedCopy
open VeriTile.Triton FP.Structural KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- A partial x-block with independent fp32 K and fp16 V storage. -/
def witnessParameters : Parameters where
  toParameters := KcacheCopyTriton.TypedCopy.witnessParameters
  V := "V"
  VCache := "vcache"
  strideVt := 64
  strideVh := 16
  strideVd := 2
  strideVcb := 256
  strideVch := 64
  strideVcs := 16
  strideVcd := 2
  headDim := 6

def witnessState : State (BitVec 32) :=
  { KcacheCopyTriton.TypedCopy.witnessState with
    mem := fun r o =>
      if r = "V" then .mk .fp16
        (if o = 104 then BitVec.ofNat 32 0x7d01 else BitVec.ofNat 32 0x8000)
      else if r = "vcache" then .mk .fp16 (BitVec.ofNat 32 0x1234)
      else KcacheCopyTriton.TypedCopy.witnessState.mem r o }

/-- Two active lanes preserve their fp32/fp16 special-value payloads; inactive
K/V destinations retain their original sentinels. This is a nonempty successful
execution of the actual masked source, with nontrivial metadata and strides. -/
theorem bit_success_witness :
    ∃ t, FP.Structural.exec (copyBackend bitsBackend)
          (source .fp32 .fp16 witnessParameters) witnessState = some t ∧
      t.mem "cache" 464 = .mk .real (BitVec.ofNat 32 0x7f800001) ∧
      t.mem "cache" 465 = .mk .real (BitVec.ofNat 32 0x80000000) ∧
      t.mem "vcache" 904 = .mk .fp16 (BitVec.ofNat 32 0x7d01) ∧
      t.mem "vcache" 906 = .mk .fp16 (BitVec.ofNat 32 0x8000) ∧
      t.mem "cache" 466 = .mk .real (BitVec.ofNat 32 0x12345678) ∧
      t.mem "vcache" 908 = .mk .fp16 (BitVec.ofNat 32 0x1234) := by
  let t := (execution bitsBackend .fp32 .fp16 witnessParameters witnessState).val
  refine ⟨t, (execution bitsBackend .fp32 .fp16 witnessParameters witnessState).property,
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have h := k_readback bitsBackend .fp32 .fp16 witnessParameters witnessState
        (⟨0, by decide⟩, PUnit.unit) (by decide) (by decide)
    simpa [witnessParameters, witnessState, KcacheCopyTriton.TypedCopy.witnessParameters,
      KcacheCopyTriton.TypedCopy.witnessState, inputOffset, outputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using h
  · have h := k_readback bitsBackend .fp32 .fp16 witnessParameters witnessState
        (⟨1, by decide⟩, PUnit.unit) (by decide) (by decide)
    simpa [witnessParameters, witnessState, KcacheCopyTriton.TypedCopy.witnessParameters,
      KcacheCopyTriton.TypedCopy.witnessState, inputOffset, outputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using h
  · have h := v_readback bitsBackend .fp32 .fp16 witnessParameters witnessState
        (⟨0, by decide⟩, PUnit.unit) (by decide)
        (fun a b _ _ hab => v_output_injective witnessParameters witnessState (by decide) hab)
    simpa [witnessParameters, witnessState, KcacheCopyTriton.TypedCopy.witnessParameters,
      KcacheCopyTriton.TypedCopy.witnessState, vInputOffset, vOutputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using h
  · have h := v_readback bitsBackend .fp32 .fp16 witnessParameters witnessState
        (⟨1, by decide⟩, PUnit.unit) (by decide)
        (fun a b _ _ hab => v_output_injective witnessParameters witnessState (by decide) hab)
    simpa [witnessParameters, witnessState, KcacheCopyTriton.TypedCopy.witnessParameters,
      KcacheCopyTriton.TypedCopy.witnessState, vInputOffset, vOutputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using h
  · have h := frame bitsBackend .fp32 .fp16 witnessParameters witnessState "cache" 466
        (Or.inr (by
          intro i hi
          simp [active, witnessParameters, KcacheCopyTriton.TypedCopy.witnessParameters] at hi
          simp [witnessParameters, witnessState, KcacheCopyTriton.TypedCopy.witnessParameters,
            KcacheCopyTriton.TypedCopy.witnessState, outputOffset, blockId, past, Region.cast, Cell.read]
          omega)) (Or.inl (by decide))
    simpa [witnessState, KcacheCopyTriton.TypedCopy.witnessState] using h
  · have h := frame bitsBackend .fp32 .fp16 witnessParameters witnessState "vcache" 908
        (Or.inl (by decide)) (Or.inr (by
          intro i hi
          simp [active, witnessParameters, KcacheCopyTriton.TypedCopy.witnessParameters] at hi
          simp [witnessParameters, witnessState, KcacheCopyTriton.TypedCopy.witnessParameters,
            KcacheCopyTriton.TypedCopy.witnessState, vOutputOffset, blockId, past, Region.cast, Cell.read]
          omega))
    simpa [witnessState] using h

#axiomsClean bit_success_witness
end
end VeriTile.Bench.TritonBenchG.KvCacheCopy.TypedCopy

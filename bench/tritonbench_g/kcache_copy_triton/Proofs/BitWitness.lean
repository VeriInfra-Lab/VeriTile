import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Total opaque backend for an executable witness. The selected unmasked copy
calls none of these numerical operations. No IEEE arithmetic implementation is
claimed by the deliberately arbitrary literal/arithmetic callbacks. -/
def bitsBackend : Algebra (BitVec 32) where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun _ _ _ _ _ => 0
  unary := fun _ _ _ => 0
  cast := fun _ _ _ x => x
  fromNat := fun _ n => BitVec.ofNat 32 n
  fromInt := fun _ n => BitVec.ofInt 32 n
  fp32Bits := fun x => x.bits
  fp32Load := id
  reduceMax := fun _ _ _ _ => fun _ => 0
  reduceSum := fun _ _ _ _ => fun _ => 0

def witnessParameters : Parameters where
  K := "K"
  KCache := "cache"
  blockTables := "tables"
  lengths := "lengths"
  splitX := 1
  strideKt := 32
  strideKh := 8
  strideKd := 1
  strideKcb := 128
  strideKch := 32
  strideKcsplit := 16
  strideKcs := 4
  strideBts := 4
  strideBtb := 1
  blockSize := 4
  width := 4

/-- Four nontrivial fp32 bit payloads: signaling NaN, negative zero, positive
infinity and the least positive subnormal. The cache starts with a sentinel. -/
def witnessState : State (BitVec 32) where
  mem := fun r o =>
    if r = "lengths" then .mk .nat 5
    else if r = "tables" then .mk .nat 3
    else if r = "K" then .mk .real
      (if o = 52 then BitVec.ofNat 32 0x7f800001
       else if o = 53 then BitVec.ofNat 32 0x80000000
       else if o = 54 then BitVec.ofNat 32 0x7f800000 else BitVec.ofNat 32 1)
    else .mk .real (BitVec.ofNat 32 0x12345678)
  regs := fun _ _ _ => none
  pids := fun a => if a = 0 then 1 else if a = 1 then 2 else 0
  numPids := fun _ => 4
  undef := fun d _ _ => defaultValue d

/-- Successful, nonempty raw-bit copy with an untouched neighboring cell.
The witness executes the actual typed source rather than a hand-written copy. -/
theorem bit_success_witness :
    ∃ t, FP.Structural.exec (copyBackend bitsBackend) (source .fp32 witnessParameters)
          witnessState = some t ∧
      t.mem "cache" 464 = .mk .real (BitVec.ofNat 32 0x7f800001) ∧
      t.mem "cache" 465 = .mk .real (BitVec.ofNat 32 0x80000000) ∧
      t.mem "cache" 466 = .mk .real (BitVec.ofNat 32 0x7f800000) ∧
      t.mem "cache" 467 = .mk .real (BitVec.ofNat 32 1) ∧
      t.mem "cache" 463 = .mk .real (BitVec.ofNat 32 0x12345678) := by
  let t := (execution bitsBackend .fp32 witnessParameters witnessState).val
  refine ⟨t, (execution bitsBackend .fp32 witnessParameters witnessState).property,
    ?_, ?_, ?_, ?_, ?_⟩
  · simpa [witnessParameters, witnessState, inputOffset, outputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using
      (readback bitsBackend .fp32 witnessParameters witnessState (⟨0, by decide⟩, PUnit.unit))
  · simpa [witnessParameters, witnessState, inputOffset, outputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using
      (readback bitsBackend .fp32 witnessParameters witnessState (⟨1, by decide⟩, PUnit.unit))
  · simpa [witnessParameters, witnessState, inputOffset, outputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using
      (readback bitsBackend .fp32 witnessParameters witnessState (⟨2, by decide⟩, PUnit.unit))
  · simpa [witnessParameters, witnessState, inputOffset, outputOffset, blockId, past,
      Storage.tag, Region.cast, Cell.read] using
      (readback bitsBackend .fp32 witnessParameters witnessState (⟨3, by decide⟩, PUnit.unit))
  · have hf := frame bitsBackend .fp32 witnessParameters witnessState "cache" 463
        (Or.inr (by
          intro i
          simp [witnessParameters, witnessState, outputOffset, blockId, past, Region.cast, Cell.read]
          omega))
    simpa [witnessState, Cell.read] using hf

#axiomsClean bitsBackend
#axiomsClean bit_success_witness
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy

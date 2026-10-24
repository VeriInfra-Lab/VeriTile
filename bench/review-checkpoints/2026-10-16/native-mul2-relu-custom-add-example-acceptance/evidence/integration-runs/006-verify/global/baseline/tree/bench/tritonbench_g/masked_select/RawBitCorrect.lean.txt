import bench.tritonbench_g.masked_select.Proofs.RawBitCopy
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (copyBackend)
open TypedCopy (Storage Parameters)
noncomputable section

/-- Five floating physical formats and one signed integer storage format. -/
inductive Format where
  | fp16 | bf16 | fp32 | f8e4 | f8e5 | int64
  deriving DecidableEq, Repr

def Format.width : Format → Nat
  | .fp16 | .bf16 => 16 | .fp32 => 32 | .f8e4 | .f8e5 => 8 | .int64 => 64

def Format.storage : Format → Storage
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32
  | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .int64 => .int64

/-- Float32 uses the actual source real channel. Integer bits are interpreted
by the exact signed toInt injection, never a floating numerical projection. -/
def payload : (f : Format) → BitVec f.width → RawCell (BitVec 32)
  | .fp16,b => .fp16 b | .bf16,b => .bf16 b | .fp32,b => .real b
  | .f8e4,b => .f8e4 b | .f8e5,b => .f8e5 b | .int64,b => .int b.toInt

@[simp] theorem payload_storage (f : Format) (b : BitVec f.width) :
    readPayload f.storage (payload f b) = payload f b := by cases f <;> rfl

theorem payload_injective (f : Format) : Function.Injective (payload f) := by
  cases f <;> intro a b h
  · exact RawCell.fp16.inj h
  · exact RawCell.bf16.inj h
  · exact RawCell.real.inj h
  · exact RawCell.f8e4.inj h
  · exact RawCell.f8e5.inj h
  · exact BitVec.eq_of_toInt_eq (RawCell.int.inj h)

/-- Complete old selected mask/prefix consumer stage, for all six finite-bit
payload formats. The sole readback condition is the old selected in-bounds
address injectivity. Prefix Nat values (including zero), initial aliases,
width/size/pids and inactive lanes keep their old unrestricted domain.
No numeric callback, no finite/nonNaN or positive-prefix assumption is added.
The Nat prefix model is NOT a proof of native int64 prefix arithmetic. -/
specification six_format_raw_bit_correctness (f : Format) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (hinj : ∀ a b, selected p m seed.pids a → selected p m seed.pids b →
      outputOffset p m seed.pids a = outputOffset p m seed.pids b → a=b) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source f.storage p) (state m seed)=some final ∧
      final.mem=encodeMemory (reference f.storage p m seed.pids) ∧
      decodeMemory final.mem=reference f.storage p m seed.pids ∧
      (∀ i : TileIndex [p.width], selected p m seed.pids i →
        final.mem p.output (outputOffset p m seed.pids i) =
        encode (readPayload f.storage (m p.input (inputOffset p seed.pids i)))) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : TileIndex [p.width], selected p m seed.pids i → o≠outputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) := complete f.storage p m seed hinj

/-- Every finite payload survives independently of all initial register/undef
contents. hinput constrains only the source cell, never the output. -/
theorem every_payload (f : Format) (p : Parameters) (m : Memory (BitVec 32))
    (seed : State (Carrier (BitVec 32))) (i : TileIndex [p.width])
    (hi : selected p m seed.pids i)
    (hinj : ∀ a b, selected p m seed.pids a → selected p m seed.pids b →
      outputOffset p m seed.pids a = outputOffset p m seed.pids b → a=b)
    (b : BitVec f.width) (hinput : m p.input (inputOffset p seed.pids i)=payload f b) :
    (TypedCopy.execution inert f.storage p (state m seed)).val.mem p.output
      (outputOffset p m seed.pids i)=encode (payload f b) := by
  simpa only [hinput,payload_storage] using readback inert f.storage p m seed i hi hinj

/-- A genuine BitVec64→signed Int→BitVec64 bridge for every pattern. This is
payload transport, not a statement about prefix producer or address overflow. -/
theorem every_int64_bits (p : Parameters) (m : Memory (BitVec 32))
    (seed : State (Carrier (BitVec 32))) (i : TileIndex [p.width])
    (hi : selected p m seed.pids i)
    (hinj : ∀ a b, selected p m seed.pids a → selected p m seed.pids b →
      outputOffset p m seed.pids a = outputOffset p m seed.pids b → a=b)
    (b : BitVec 64) (hinput : m p.input (inputOffset p seed.pids i)=RawCell.int b.toInt) :
    BitVec.ofInt 64 (((TypedCopy.execution inert .int64 p (state m seed)).val.mem p.output
      (outputOffset p m seed.pids i)).read .int)=b := by
  have h := every_payload .int64 p m seed i hi hinj b hinput
  simp only [Format.storage,payload] at h
  rw [h]
  simp only [encode,Cell.read_same]
  exact BitVec.ofInt_toInt

/-- Preserve all seven legacy source annotations and arbitrary legacy carrier
R separately. fp64 remains erased real, not a native floating64 format.
No old abstract source theorem is replaced or narrowed. -/
theorem legacy_raw_carrier_correctness {R : Type} [Inhabited R] (d : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R))
    (hinj : ∀ a b, selected p m seed.pids a → selected p m seed.pids b →
      outputOffset p m seed.pids a = outputOffset p m seed.pids b → a=b) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source d p) (state m seed)=some final ∧
      final.mem=encodeMemory (reference d p m seed.pids) ∧
      decodeMemory final.mem=reference d p m seed.pids ∧
      (∀ i : TileIndex [p.width], selected p m seed.pids i →
        final.mem p.output (outputOffset p m seed.pids i) =
        encode (readPayload d (m p.input (inputOffset p seed.pids i)))) ∧
      (∀ r o, (r≠p.output ∨ ∀ i : TileIndex [p.width], selected p m seed.pids i → o≠outputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) := complete d p m seed hinj

#axiomsClean payload_storage
#axiomsClean payload_injective
#axiomsClean six_format_raw_bit_correctness
#axiomsClean every_payload
#axiomsClean every_int64_bits
#axiomsClean legacy_raw_carrier_correctness
#axiomsClean TypedCopy.floating_source_projection
#axiomsClean TypedCopy.source_control
#axiomsClean TypedCopy.int64_payload_dtype
end
end VeriTile.Bench.TritonBenchG.MaskedSelect.RawBits

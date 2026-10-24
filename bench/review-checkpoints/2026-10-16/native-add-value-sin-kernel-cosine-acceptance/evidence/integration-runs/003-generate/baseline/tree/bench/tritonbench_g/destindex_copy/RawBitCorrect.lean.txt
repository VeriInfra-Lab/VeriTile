import bench.tritonbench_g.destindex_copy.Proofs.RawBitCopy
namespace VeriTile.Bench.TritonBenchG.DestindexCopy.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (copyBackend)
open TypedSource (Storage)
noncomputable section

/-- All nine independently chosen fp32/fp16/bf16 pairs of the complete original
source. Each input/output pair shares its storage dtype; there is no floating
arithmetic or cross-dtype conversion. All word payloads, signed Int metadata,
Nat dimensions/strides/program ids, unmasked padding, aliases, initial registers
and undef values are retained. Readback uses only the old injectivity/separate
output conditions; whole-memory and frame lemmas need neither condition.
This proves a software word-memory model, not fixed-width integer addressing,
byte ABI, memory safety, cross-program race freedom or CUDA conformance. -/
specification nine_storage_pair_raw_bit_correctness (nd rd : Storage) (p : Layout)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (hn : Function.Injective (outputNope p m seed.pids))
    (hr : Function.Injective (outputRope p m seed.pids)) (hsep : p.O_nope ≠ p.O_rope) :
    ∃ final, FP.Structural.exec (copyBackend inert) (p.source nd rd Bool.false) (state m seed)=some final ∧
      final.mem=encodeMemory (reference nd rd p m seed.pids) ∧
      decodeMemory final.mem=reference nd rd p m seed.pids ∧
      (∀ i : TileIndex [1,p.BLOCK_DMODEL_NOPE], final.mem p.O_nope (outputNope p m seed.pids i) =
        encode (readStorage (storage nd) (m p.KV_nope (inputNope p seed.pids i)))) ∧
      (∀ i : TileIndex [1,p.BLOCK_DMODEL_ROPE], final.mem p.O_rope (outputRope p m seed.pids i) =
        encode (readStorage (storage rd) (m p.KV_rope (inputRope p seed.pids i)))) ∧
      (∀ r o,
        (r≠p.O_nope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_NOPE], o≠outputNope p m seed.pids i) →
        (r≠p.O_rope ∨ ∀ i : TileIndex [1,p.BLOCK_DMODEL_ROPE], o≠outputRope p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) :=
  complete nd rd p m seed hn hr hsep

def width : Storage → Nat
  | .fp32 => 32 | .fp16 | .bf16 => 16

def payload : (d : Storage) → BitVec (width d) → RawCell (BitVec 32)
  | .fp32, b => .real b | .fp16, b => .fp16 b | .bf16, b => .bf16 b

@[simp] theorem payload_storage (d : Storage) (b : BitVec (width d)) :
    readStorage (storage d) (payload d b) = payload d b := by cases d <;> rfl

theorem payload_injective (d : Storage) : Function.Injective (payload d) := by
  cases d <;> intro a b h <;> cases h <;> rfl

/-- No finite/nonNaN input premise: every raw bit pattern is preserved. -/
theorem every_nope_payload (nd rd : Storage) (p : Layout)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (i : TileIndex [1,p.BLOCK_DMODEL_NOPE])
    (hinj : Function.Injective (outputNope p m seed.pids)) (hsep : p.O_nope ≠ p.O_rope)
    (b : BitVec (width nd)) (hinput : m p.KV_nope (inputNope p seed.pids i)=payload nd b) :
    (TypedBits.execution inert nd rd p (state m seed)).val.mem p.O_nope
      (outputNope p m seed.pids i) = encode (payload nd b) := by
  simpa only [hinput,payload_storage] using nope_readback inert nd rd p m seed i hinj hsep

/-- ROPE is loaded before NOPE is stored, even when NOPE output aliases ROPE input. -/
theorem every_rope_payload (nd rd : Storage) (p : Layout)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (i : TileIndex [1,p.BLOCK_DMODEL_ROPE])
    (hinj : Function.Injective (outputRope p m seed.pids))
    (b : BitVec (width rd)) (hinput : m p.KV_rope (inputRope p seed.pids i)=payload rd b) :
    (TypedBits.execution inert nd rd p (state m seed)).val.mem p.O_rope
      (outputRope p m seed.pids i) = encode (payload rd b) := by
  simpa only [hinput,payload_storage] using rope_readback inert nd rd p m seed i hinj

#axiomsClean nine_storage_pair_raw_bit_correctness
#axiomsClean payload_storage
#axiomsClean payload_injective
#axiomsClean every_nope_payload
#axiomsClean every_rope_payload
end
end VeriTile.Bench.TritonBenchG.DestindexCopy.RawBits

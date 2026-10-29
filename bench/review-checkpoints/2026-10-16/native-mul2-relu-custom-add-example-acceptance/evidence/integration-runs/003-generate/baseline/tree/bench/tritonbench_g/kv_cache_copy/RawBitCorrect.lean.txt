import bench.tritonbench_g.kv_cache_copy.Proofs.RawBitCopy
namespace VeriTile.Bench.TritonBenchG.KvCacheCopy.RawBits
open VeriTile.Triton FP.Structural
open KcacheCopyTriton.RawBits
open KcacheCopyTriton.TypedCopy (Storage copyBackend)
open TypedCopy (Parameters)
noncomputable section

/-- All 25 mixed pairs of the five width-explicit formats, in the complete
previously accepted masked x-block. K and V may have different formats but
input/cache formats match within each pair. Only the old separate-cache and
active V address injectivity conditions are required for both readbacks.
There is no numerical callback, finite/nonNaN restriction, positive stride or
positive width premise. All raw memory cells and initial registers/undef are
allowed. The memory is word-addressed; no CUDA/byte-ABI conformance is asserted. -/
specification mixed_five_format_raw_bit_correctness (fk fv : MachineFormat) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32))) (hcache : p.KCache ≠ p.VCache)
    (hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b → vOutputOffset p m seed.pids a = vOutputOffset p m seed.pids b → a=b) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source fk.storage fv.storage p) (state m seed)=some final ∧
      final.mem=encodeMemory (reference fk.storage fv.storage p m seed.pids) ∧
      decodeMemory final.mem=reference fk.storage fv.storage p m seed.pids ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i → final.mem p.KCache (outputOffset p.toParameters m seed.pids i) =
        encode (readStorage fk.storage (m p.K (inputOffset p.toParameters seed.pids i)))) ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i → final.mem p.VCache (vOutputOffset p m seed.pids i) =
        encode (readStorage fv.storage (m p.V (vInputOffset p seed.pids i)))) ∧
      (∀ r o,
        (r≠p.KCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o≠outputOffset p.toParameters m seed.pids i) →
        (r≠p.VCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o≠vOutputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) :=
  complete fk.storage fv.storage p m seed hcache hinj

/-- Every K payload survives, including arbitrary NaN encodings and signed
zeros. hinput specifies only the independent initial memory. -/
theorem every_k_payload (fk fv : MachineFormat) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (i : TileIndex [p.width]) (hi : TypedCopy.active p i) (hcache : p.KCache ≠ p.VCache)
    (b : BitVec fk.width) (hinput : m p.K (inputOffset p.toParameters seed.pids i)=payload fk b) :
    (TypedCopy.execution inert fk.storage fv.storage p (state m seed)).val.mem p.KCache
      (outputOffset p.toParameters m seed.pids i) = encode (payload fk b) := by
  simpa only [hinput,payload_storage] using k_readback inert fk.storage fv.storage p m seed i hi hcache

/-- V also reads its initial payload even if K stores alias its input region. -/
theorem every_v_payload (fk fv : MachineFormat) (p : Parameters)
    (m : Memory (BitVec 32)) (seed : State (Carrier (BitVec 32)))
    (i : TileIndex [p.width]) (hi : TypedCopy.active p i)
    (hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b → vOutputOffset p m seed.pids a = vOutputOffset p m seed.pids b → a=b)
    (b : BitVec fv.width) (hinput : m p.V (vInputOffset p seed.pids i)=payload fv b) :
    (TypedCopy.execution inert fk.storage fv.storage p (state m seed)).val.mem p.VCache
      (vOutputOffset p m seed.pids i) = encode (payload fv b) := by
  simpa only [hinput,payload_storage] using v_readback inert fk.storage fv.storage p m seed i hi hinj

/-- All 36 legacy source-annotation combinations are separately retained over
arbitrary R. In particular the annotation named fp64 remains an abstract real
channel, NOT a native64 format. Both real channels share R as in the original
abstract evaluator; finite raw/legacy memory is not identified with arbitrary
noncanonical Structural cells. No old theorem or source annotation is changed. -/
theorem legacy_raw_carrier_correctness {R : Type} [Inhabited R] (kd vd : Storage) (p : Parameters)
    (m : Memory R) (seed : State (Carrier R)) (hcache : p.KCache ≠ p.VCache)
    (hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b → vOutputOffset p m seed.pids a = vOutputOffset p m seed.pids b → a=b) :
    ∃ final, FP.Structural.exec (copyBackend inert) (TypedCopy.source kd vd p) (state m seed)=some final ∧
      final.mem=encodeMemory (reference kd vd p m seed.pids) ∧
      decodeMemory final.mem=reference kd vd p m seed.pids ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i → final.mem p.KCache (outputOffset p.toParameters m seed.pids i) =
        encode (readStorage kd (m p.K (inputOffset p.toParameters seed.pids i)))) ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i → final.mem p.VCache (vOutputOffset p m seed.pids i) =
        encode (readStorage vd (m p.V (vInputOffset p seed.pids i)))) ∧
      (∀ r o,
        (r≠p.KCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o≠outputOffset p.toParameters m seed.pids i) →
        (r≠p.VCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o≠vOutputOffset p m seed.pids i) →
        final.mem r o=encode (m r o)) ∧
      (∀ r o, Canonical (final.mem r o)) :=
  complete kd vd p m seed hcache hinj

#axiomsClean mixed_five_format_raw_bit_correctness
#axiomsClean every_k_payload
#axiomsClean every_v_payload
#axiomsClean legacy_raw_carrier_correctness
#axiomsClean TypedCopy.source_projection
end
end VeriTile.Bench.TritonBenchG.KvCacheCopy.RawBits

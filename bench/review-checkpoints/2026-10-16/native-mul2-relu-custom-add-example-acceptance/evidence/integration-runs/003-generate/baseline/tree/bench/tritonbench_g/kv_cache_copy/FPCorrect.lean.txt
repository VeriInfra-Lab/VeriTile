import bench.tritonbench_g.kv_cache_copy.Proofs.TypedCopy
import bench.tritonbench_g.kv_cache_copy.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.KvCacheCopy
open VeriTile.Triton FP.Structural KcacheCopyTriton.TypedCopy TypedCopy
noncomputable section

/-- Typed execution of the accepted x-block, with two independent input
snapshots. Hypotheses are the existing active-address/caches conditions;
there is no floating numerical law or assumed output value. -/
specification kv_cache_copy_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (kd vd : Storage) (p : TypedCopy.Parameters) (s : State α)
    (hcache : p.KCache ≠ p.VCache)
    (hinj : ∀ a b, TypedCopy.active p a → TypedCopy.active p b →
      vOutputOffset p s a = vOutputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (TypedCopy.source kd vd p) s = some t ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i →
        t.mem p.KCache (outputOffset p.toParameters s i) =
          .mk kd.tag ((s.mem p.K (inputOffset p.toParameters s i)).read kd.tag)) ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p i →
        t.mem p.VCache (vOutputOffset p s i) =
          .mk vd.tag ((s.mem p.V (vInputOffset p s i)).read vd.tag)) ∧
      (∀ r o,
        (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o ≠ outputOffset p.toParameters s i) →
        (r ≠ p.VCache ∨ ∀ i : TileIndex [p.width], TypedCopy.active p i → o ≠ vOutputOffset p s i) →
        t.mem r o = s.mem r o) :=
  TypedCopy.correctness rest kd vd p s hcache hinj

/-- A partial tile successfully copies distinct fp32/fp16 payloads and leaves
both masked-off destinations unchanged. -/
theorem kv_cache_copy_bit_success :
    ∃ t, FP.Structural.exec (copyBackend bitsBackend)
          (TypedCopy.source .fp32 .fp16 TypedCopy.witnessParameters) TypedCopy.witnessState = some t ∧
      t.mem "cache" 464 = .mk .real (BitVec.ofNat 32 0x7f800001) ∧
      t.mem "cache" 465 = .mk .real (BitVec.ofNat 32 0x80000000) ∧
      t.mem "vcache" 904 = .mk .fp16 (BitVec.ofNat 32 0x7d01) ∧
      t.mem "vcache" 906 = .mk .fp16 (BitVec.ofNat 32 0x8000) ∧
      t.mem "cache" 466 = .mk .real (BitVec.ofNat 32 0x12345678) ∧
      t.mem "vcache" 908 = .mk .fp16 (BitVec.ofNat 32 0x1234) :=
  TypedCopy.bit_success_witness

#axiomsClean kv_cache_copy_typed_correctness
#axiomsClean kv_cache_copy_bit_success
end
end VeriTile.Bench.TritonBenchG.KvCacheCopy

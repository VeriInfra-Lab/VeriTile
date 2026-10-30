import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy
import bench.tritonbench_g.kcache_copy_triton.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton
open VeriTile.Triton FP.Structural TypedCopy
noncomputable section

/-- Successful raw typed copy for the accepted n_tokens=1 split-x source.
All source metadata/address parameters remain explicit in `Parameters`.
No floating arithmetic, rounding, or finite-value law is assumed. -/
specification kcache_copy_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (d : Storage) (p : Parameters) (s : State α) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], t.mem p.KCache (outputOffset p s i) =
        .mk d.tag ((s.mem p.K (inputOffset p s i)).read d.tag)) ∧
      (∀ r o, (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p s i) →
        t.mem r o = s.mem r o) :=
  TypedCopy.correctness rest d p s

/-- Nonempty typed fp32 execution preserves signaling-NaN, signed-zero,
infinity and subnormal payloads, plus a neighboring untouched cell. -/
theorem kcache_copy_bit_success :
    ∃ t, FP.Structural.exec (copyBackend bitsBackend) (source .fp32 witnessParameters)
          witnessState = some t ∧
      t.mem "cache" 464 = .mk .real (BitVec.ofNat 32 0x7f800001) ∧
      t.mem "cache" 465 = .mk .real (BitVec.ofNat 32 0x80000000) ∧
      t.mem "cache" 466 = .mk .real (BitVec.ofNat 32 0x7f800000) ∧
      t.mem "cache" 467 = .mk .real (BitVec.ofNat 32 1) ∧
      t.mem "cache" 463 = .mk .real (BitVec.ofNat 32 0x12345678) :=
  TypedCopy.bit_success_witness

#axiomsClean kcache_copy_typed_correctness
#axiomsClean kcache_copy_bit_success
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton

import bench.tritonbench_g.masked_select.Proofs.TypedCopy
import bench.tritonbench_g.masked_select.Proofs.BitWitness

namespace VeriTile.Bench.TritonBenchG.MaskedSelect
open VeriTile.Triton FP.Structural TypedCopy
open KcacheCopyTriton.TypedCopy (copyBackend bitsBackend)
noncomputable section

/-- Actual typed selection-copy with successful execution, independent input
readback and frame. Only active output addresses must be collision-free. -/
specification masked_select_typed_correctness {α : Type} [Inhabited α]
    (rest : Algebra α) (d : Storage) (p : Parameters) (s : State α)
    (hinj : ∀ a b, TypedCopy.active p s a → TypedCopy.active p s b →
      outputOffset p s a = outputOffset p s b → a = b) :
    ∃ t, FP.Structural.exec (copyBackend rest) (source d p) s = some t ∧
      (∀ i : TileIndex [p.width], TypedCopy.active p s i →
        t.mem p.output (outputOffset p s i) =
          .mk d.tag ((s.mem p.input (inputOffset p s i)).read d.tag)) ∧
      (∀ r o, (r ≠ p.output ∨ ∀ i : TileIndex [p.width], TypedCopy.active p s i →
        o ≠ outputOffset p s i) → t.mem r o = s.mem r o) :=
  TypedCopy.correctness rest d p s hinj

/-- All signed 64-bit input patterns survive the actual int64 source path. -/
theorem masked_select_int64_bits {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Parameters) (s : State α) (i : TileIndex [p.width]) (hi : TypedCopy.active p s i)
    (hinj : ∀ a b, TypedCopy.active p s a → TypedCopy.active p s b →
      outputOffset p s a = outputOffset p s b → a = b)
    (bits : BitVec 64) (htyped : s.mem p.input (inputOffset p s i) = .mk .int bits.toInt) :
    BitVec.ofInt 64 (((execution rest .int64 p s).val.mem p.output (outputOffset p s i)).read .int) = bits :=
  TypedCopy.int64_bits_readback rest p s i hi hinj bits htyped

theorem masked_select_bit_success :
    ∃ t, FP.Structural.exec (copyBackend bitsBackend) (source .fp32 witnessParameters)
          witnessState = some t ∧
      t.mem "output" 1 = .mk .real (BitVec.ofNat 32 0x7f800001) ∧
      t.mem "output" 2 = .mk .real (BitVec.ofNat 32 0x80000000) ∧
      t.mem "output" 0 = .mk .real (BitVec.ofNat 32 0x12345678) ∧
      t.mem "output" 3 = .mk .real (BitVec.ofNat 32 0x12345678) :=
  TypedCopy.bit_success_witness

#axiomsClean masked_select_typed_correctness
#axiomsClean masked_select_int64_bits
#axiomsClean masked_select_bit_success
end
end VeriTile.Bench.TritonBenchG.MaskedSelect

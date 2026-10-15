import bench.tritonbench_g.kcache_copy_triton.Proofs.SignedDiscreteCopy

namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.DiscreteCorrect
open VeriTile.Triton FP.Structural SignedDiscreteCopy
open SignedCopy (Parameters inputOffset outputOffset)
noncomputable section

def width : Storage → Nat
  | .i8 | .u8 => 8
  | .i16 | .u16 => 16
  | .i32 | .u32 => 32
  | .i64 | .u64 => 64
  | .boolean => 1

/-- Explicit finite storage representations; signed words retain their sign
and boolean storage has exactly the two source values. -/
def payloadValue {α : Type} : (d : Storage) → BitVec (width d) → Value α d.tag
  | .i8, b => b.toInt | .i16, b => b.toInt
  | .i32, b => b.toInt | .i64, b => b.toInt
  | .u8, b => b.toNat | .u16, b => b.toNat
  | .u32, b => b.toNat | .u64, b => b.toNat
  | .boolean, b => b != 0

def payload {α : Type} (d : Storage) (b : BitVec (width d)) : Cell α :=
  .mk d.tag (payloadValue d b)

theorem payloadValue_injective {α : Type} (d : Storage) :
    Function.Injective (@payloadValue α d) := by
  cases d <;> intro a b h
  all_goals try exact BitVec.toInt_inj.mp h
  all_goals try exact BitVec.eq_of_toNat_eq h
  · change (a != 0) = (b != 0) at h
    have boolean_injective : ∀ a b : BitVec 1, (a != 0) = (b != 0) → a = b := by
      decide +kernel
    exact boolean_injective a b h

/-- Nine discrete data-storage profiles of the complete multi-token copy.
No floating arithmetic callback or numerical law occurs in the result.
Metadata is the existing signed logical source model; native fixed-width
address lowering, storage ABI and host composition remain separate. -/
specification all_tokens_discrete_correctness
    {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (initial : State α) :
    ∃ final, FP.Structural.exec (TypedCopy.copyBackend rest) (source d p) initial = some final ∧
      final.mem = (copied d p initial).mem ∧
      (∀ i : TileIndex [p.width], final.mem p.KCache (outputOffset p initial i) = loaded d p initial i) ∧
      (∀ r o, (r ≠ p.KCache ∨ ∀ i : TileIndex [p.width], o ≠ outputOffset p initial i) →
        final.mem r o = initial.mem r o) := correctness rest d p initial

/-- Every finite payload is preserved injectively, including negative signed
integers, unsigned high-bit values and both boolean values. -/
theorem every_discrete_payload
    {α : Type} [Inhabited α] (rest : Algebra α) (d : Storage)
    (p : Parameters) (initial : State α) (i : TileIndex [p.width])
    (b : BitVec (width d))
    (htyped : initial.mem p.K (inputOffset p initial i) = payload d b) :
    (execution rest d p initial).val.mem p.KCache (outputOffset p initial i) = payload d b :=
  typed_readback rest d p initial i (payloadValue d b) htyped

#axiomsClean payloadValue_injective
#axiomsClean all_tokens_discrete_correctness
#axiomsClean every_discrete_payload
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.DiscreteCorrect

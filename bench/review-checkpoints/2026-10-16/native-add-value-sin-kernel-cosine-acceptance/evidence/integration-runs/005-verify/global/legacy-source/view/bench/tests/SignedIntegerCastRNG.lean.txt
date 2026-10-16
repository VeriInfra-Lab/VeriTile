import VeriTile.Triton.Float.SignedCast
import VeriTile.Meta.StatementAudit

/-! The source offset range is established by the actual narrowing operation.
No bound on the original mathematical Int offsets is assumed. -/
namespace VeriTile.Tests.SignedIntegerCastRNG
open VeriTile.Triton

theorem narrowed_in_range (x : Int) :
    Random.IntegerType.inRange .i32 (BitVec.ofInt 32 x).toInt = Bool.true := by
  simp only [Random.IntegerType.inRange, decide_eq_true_eq]
  exact ⟨by simpa using BitVec.le_toInt (BitVec.ofInt 32 x),
    by simpa using (BitVec.toInt_lt (x := BitVec.ofInt 32 x))⟩

theorem narrowed_tensor_valid {sh : TileShape}
    (vs vo : TileIndex sh → Int) (rounds : Nat)
    (seeds_valid : ∀ i, Random.IntegerType.inRange .i64 (vs i) = Bool.true) :
    (⟨.i64, .i32, rounds⟩ : Random.SourceProfile).tensorInputsValid vs
      (fun i => (BitVec.ofInt 32 (vo i)).toInt) = Bool.true := by
  rw [Random.SourceProfile.tensorInputsValid_iff]
  exact ⟨seeds_valid, fun i => narrowed_in_range (vo i)⟩

theorem typed_narrowed_random {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) {sh : TileShape}
    (seeds offsets : Op .int sh) (s : FP.Structural.State α)
    (vs vo : TileIndex sh → Int) (rounds : Nat)
    (hs : FP.Structural.evalOp M p seeds s = some vs)
    (ho : FP.Structural.evalOp M p offsets s = some vo)
    (seeds_valid : ∀ i, Random.IntegerType.inRange .i64 (vs i) = Bool.true) :
    FP.Structural.evalOp M p
      (.randTensor ⟨.i64, .i32, rounds⟩ seeds (Op.intToSigned 32 offsets)) s =
      some (fun i => M.fp32Bits
        ⟨(Random.rand (vs i) (BitVec.ofInt 32 (vo i)).toInt .bits32 rounds).bits⟩) := by
  rw [FP.Structural.evalOp, FP.Structural.eval_intToSigned, hs, ho]
  simp only [bind, pure, Option.bind_some]
  rw [narrowed_tensor_valid vs vo rounds seeds_valid]
  rfl

#axiomsClean narrowed_in_range
#axiomsClean narrowed_tensor_valid
#axiomsClean typed_narrowed_random
end VeriTile.Tests.SignedIntegerCastRNG

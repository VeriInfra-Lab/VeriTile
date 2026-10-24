import VeriTile.Triton.Core.SignedCast
import VeriTile.Triton.Float.FixedInteger

/-! Signed casts in all three evaluators. These laws cover arbitrary Int
inputs, including negative values and values outside the destination range. -/
namespace VeriTile.Triton

theorem signedBits_int_mod (width : Nat) (value : Int) :
    IntegerExpr.signedBits width (value % (2^width : Int)).toNat =
      (BitVec.ofInt width value).toInt := by
  rw [signedBits_eq_bitvec]
  congr 1
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat, BitVec.toNat_ofInt, Nat.cast_pow, Nat.cast_ofNat]
  exact Nat.mod_eq_of_lt (unsigned_cast_lt width value)

theorem eval_intToSigned (width : Nat) (value : Op .int shape) (s : BlockState) :
    evalOp (Op.intToSigned width value) s = (do
      let t ← evalOp value s
      pure ⟨fun i => (BitVec.ofInt width (t.data i)).toInt⟩) := by
  rw [Op.intToSigned, eval_unsignedToInt, eval_intToUnsigned]
  cases h : evalOp value s <;>
    simp only [h, bind, pure, Option.bind_none, Option.bind_some]
  congr 1
  ext i
  exact signedBits_int_mod width _

theorem evalOpR_intToSigned (R : RoundingModel) (width : Nat)
    (value : Op .int shape) (s : BlockState) :
    evalOpR R (Op.intToSigned width value) s = (do
      let t ← evalOpR R value s
      pure ⟨fun i => (BitVec.ofInt width (t.data i)).toInt⟩) := by
  rw [Op.intToSigned, evalOpR_unsignedToInt, evalOpR_intToUnsigned]
  cases h : evalOpR R value s <;>
    simp only [h, bind, pure, Option.bind_none, Option.bind_some]
  congr 1
  ext i
  exact signedBits_int_mod width _

theorem FP.Structural.eval_intToSigned {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (width : Nat)
    (value : Op .int shape) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.intToSigned width value) s = (do
      let t ← FP.Structural.evalOp M p value s
      pure (fun i => (BitVec.ofInt width (t i)).toInt)) := by
  rw [Op.intToSigned, FP.Structural.eval_unsignedToInt, FP.Structural.eval_intToUnsigned]
  cases h : FP.Structural.evalOp M p value s <;>
    simp only [h, bind, pure, Option.bind_none, Option.bind_some]
  apply congrArg some
  funext i
  exact signedBits_int_mod width _

theorem eval_boolToInt (value : Op .bool shape) (s : BlockState) :
    evalOp (Op.boolToInt value) s = (do
      let t ← evalOp value s
      pure ⟨fun i => if t.data i then (1 : Int) else 0⟩) := by
  cases h : evalOp value s <;>
    simp only [Op.boolToInt, evalOp, h, bind, pure, Option.bind_none,
      Option.bind_some, Tile.select, Tile.scalar]

theorem evalOpR_boolToInt (R : RoundingModel) (value : Op .bool shape) (s : BlockState) :
    evalOpR R (Op.boolToInt value) s = (do
      let t ← evalOpR R value s
      pure ⟨fun i => if t.data i then (1 : Int) else 0⟩) := by
  cases h : evalOpR R value s <;>
    simp only [Op.boolToInt, evalOpR, h, bind, pure, Option.bind_none,
      Option.bind_some, Tile.select, Tile.scalar]

theorem FP.Structural.eval_boolToInt {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType)
    (value : Op .bool shape) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.boolToInt value) s = (do
      let t ← FP.Structural.evalOp M p value s
      pure (fun i => if t i then (1 : Int) else 0)) := by
  cases h : FP.Structural.evalOp M p value s <;>
    simp only [Op.boolToInt, FP.Structural.evalOp, h, bind, pure,
      Option.bind_none, Option.bind_some]

end VeriTile.Triton

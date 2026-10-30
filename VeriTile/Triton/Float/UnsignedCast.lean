import VeriTile.Triton.Core.Integer
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.EvalOpR

/-! Unsigned machine casts in the real, rounded and structural evaluators.
The cast keeps low bits independently of the floating backend. Source integer
representability is an input contract, not inferred from the cast's output. -/
namespace VeriTile.Triton

theorem eval_intToUnsigned (width : Nat) (value : Op .int shape) (s : BlockState) :
    evalOp (Op.intToUnsigned width value) s = (do
      let t ← evalOp value s
      pure ⟨fun i => (t.data i % (2^width : Int)).toNat⟩) := by
  cases h : evalOp value s <;>
    simp only [Op.intToUnsigned, evalOp, h, bind, pure, Option.bind_none, Option.bind_some,
      Tile.bop, Tile.scalar, IntegralDType.mod, IntegerExpr.same_left, IntegerExpr.same_right]

theorem evalOpR_intToUnsigned (R : RoundingModel) (width : Nat)
    (value : Op .int shape) (s : BlockState) :
    evalOpR R (Op.intToUnsigned width value) s = (do
      let t ← evalOpR R value s
      pure ⟨fun i => (t.data i % (2^width : Int)).toNat⟩) := by
  cases h : evalOpR R value s <;>
    simp only [Op.intToUnsigned, evalOpR, h, bind, pure, Option.bind_none, Option.bind_some,
      Tile.bop, Tile.scalar, IntegralDType.mod, IntegerExpr.same_left, IntegerExpr.same_right]

theorem FP.Structural.eval_intToUnsigned {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (width : Nat)
    (value : Op .int shape) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.intToUnsigned width value) s = (do
      let t ← FP.Structural.evalOp M p value s
      pure (fun i => (t i % (2^width : Int)).toNat)) := by
  cases h : FP.Structural.evalOp M p value s <;>
    simp only [FP.Structural.evalOp, h, bind, pure,
      Option.bind_none, Option.bind_some, FP.Structural.bop, FP.Structural.integralMod,
      IntegerExpr.same_left]

theorem unsigned_cast_lt (width : Nat) (value : Int) :
    (value % (2^width : Int)).toNat < 2^width := by
  have hm : (0 : Int) < 2^width := pow_pos (by decide) _
  have hlo := Int.emod_nonneg value (ne_of_gt hm)
  have hhi := Int.emod_lt_of_pos value hm
  apply (Int.toNat_lt hlo).mpr
  exact_mod_cast hhi

theorem eval_natToUnsigned (width : Nat) (value : Op .nat shape) (s : BlockState) :
    evalOp (Op.natToUnsigned width value) s = (do
      let t ← evalOp value s
      pure ⟨fun i => t.data i % 2^width⟩) := by
  cases h : evalOp value s <;>
    simp only [Op.natToUnsigned, evalOp, h, bind, pure, Option.bind_none, Option.bind_some,
      Tile.bop, Tile.scalar, IntegralDType.mod, IntegerExpr.same_left, IntegerExpr.same_right]

theorem FP.Structural.eval_natToUnsigned {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (width : Nat)
    (value : Op .nat shape) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.natToUnsigned width value) s = (do
      let t ← FP.Structural.evalOp M p value s
      pure (fun i => t i % 2^width)) := by
  cases h : FP.Structural.evalOp M p value s with
  | none => simp only [FP.Structural.evalOp, h, bind, Option.bind_none]
  | some t =>
      simp only [FP.Structural.evalOp, h, bind, pure, Option.bind_some]
      apply congrArg some
      funext i
      change t ((IntegerExpr.sameShape shape).leftIndex i) % 2^width = t i % 2^width
      rw [IntegerExpr.same_left]

theorem unsigned_cast_of_nonneg_lt (width : Nat) (value : Int)
    (hlo : 0 ≤ value) (hhi : value < 2^width) :
    (value % (2^width : Int)).toNat = value.toNat := by
  rw [Int.emod_eq_of_lt hlo hhi]

/-- The formula is precisely the unsigned interpretation of the low bits. -/
theorem unsigned_cast_eq_bitvec (width : Nat) (value : Int) :
    (value % (2^width : Int)).toNat = (BitVec.ofInt width value).toNat := by
  simp only [BitVec.toNat_ofInt, Nat.cast_pow, Nat.cast_ofNat]

theorem unsigned_cast_minus_one : ((-1 : Int) % 2^32).toNat = 4294967295 := by decide
theorem unsigned_cast_high_bits : ((2^32 + 7 : Int) % 2^32).toNat = 7 := by decide
theorem unsigned_cast_min_int64 : ((-(2^63) : Int) % 2^32).toNat = 0 := by decide

end VeriTile.Triton

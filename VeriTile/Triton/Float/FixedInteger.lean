import VeriTile.Triton.Core.FixedInteger
import VeriTile.Triton.Float.UnsignedCast

/-! Evaluator and bit-vector meaning of the derived signed integer operations.
These laws are independent of the floating backend. -/
namespace VeriTile.Triton

def IntegerExpr.signedBits (width value : Nat) : Int :=
  let bits := value % 2^width
  if 2 * bits < 2^width then (bits : Int) else (bits : Int) - 2^width

theorem signedBits_eq_bitvec (width value : Nat) :
    IntegerExpr.signedBits width value = (BitVec.ofNat width value).toInt := by
  simp only [IntegerExpr.signedBits, BitVec.toInt_eq_toNat_cond, BitVec.toNat_ofNat,
    Nat.cast_pow, Nat.cast_ofNat]

theorem eval_unsignedToInt (width : Nat) (value : Op .nat shape) (s : BlockState) :
    evalOp (Op.unsignedToInt width value) s = (do
      let t ← evalOp value s
      pure ⟨fun i => IntegerExpr.signedBits width (t.data i)⟩) := by
  cases h : evalOp value s <;>
    simp only [Op.unsignedToInt, Op.natToUnsigned, evalOp, h, bind, pure,
      Option.bind_none, Option.bind_some, Tile.bop, Tile.cop, Tile.scalar, Tile.select,
      IntegralDType.mod, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits,
      decide_eq_true_eq]
  rfl

theorem eval_intShiftRight (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .nat b) (s : BlockState) :
    evalOp (Op.intShiftRight bc lhs rhs) s = (do
      let x ← evalOp lhs s
      let y ← evalOp rhs s
      pure ⟨fun i => x.data (bc.leftIndex i) >>> y.data (bc.rightIndex i)⟩) := by
  cases hx : evalOp lhs s <;> cases hy : evalOp rhs s <;>
    simp only [Op.intShiftRight, evalOp, hx, hy, bind, pure,
      Option.bind_none, Option.bind_some, Tile.bop, Tile.scalar,
      IntegralDType.floorDiv, IntegerExpr.same_left, IntegerExpr.same_right]
  congr 1
  ext i
  simp only [Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, one_mul,
    Int.ofNat_eq_natCast, Nat.cast_pow, Nat.cast_ofNat]

theorem eval_intBitAnd (width : Nat) (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : BlockState) :
    evalOp (Op.intBitAnd width bc lhs rhs) s = (do
      let x ← evalOp lhs s
      let y ← evalOp rhs s
      pure ⟨fun i => ((BitVec.ofInt width (x.data (bc.leftIndex i))) &&&
        (BitVec.ofInt width (y.data (bc.rightIndex i)))).toInt⟩) := by
  rw [Op.intBitAnd, eval_unsignedToInt]
  simp only [evalOp, eval_intToUnsigned]
  cases hx : evalOp lhs s <;> cases hy : evalOp rhs s <;>
    simp only [hx, hy, bind, pure, Option.bind_none, Option.bind_some]
  congr 1
  ext i
  dsimp only
  rw [signedBits_eq_bitvec]
  congr 1
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat, BitVec.toNat_and, BitVec.toNat_ofInt,
    Tile.bop, Nat.cast_pow, Nat.cast_ofNat]
  apply Nat.mod_eq_of_lt
  exact lt_of_le_of_lt (Nat.and_le_left) (unsigned_cast_lt width _)

/-- Integer operations do not depend on the rounding model. -/
theorem evalOpR_unsignedToInt (R : RoundingModel) (width : Nat)
    (value : Op .nat shape) (s : BlockState) :
    evalOpR R (Op.unsignedToInt width value) s = (do
      let t ← evalOpR R value s
      pure ⟨fun i => IntegerExpr.signedBits width (t.data i)⟩) := by
  cases h : evalOpR R value s <;>
    simp only [Op.unsignedToInt, Op.natToUnsigned, evalOpR, h, bind, pure,
      Option.bind_none, Option.bind_some, Tile.bop, Tile.cop, Tile.scalar, Tile.select,
      IntegralDType.mod, NumericDType.mul, NumericDType.sub, ComparableDType.lt,
      IntegerExpr.same_left, IntegerExpr.same_right, IntegerExpr.signedBits,
      decide_eq_true_eq]
  rfl

theorem evalOpR_intShiftRight (R : RoundingModel) (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .nat b) (s : BlockState) :
    evalOpR R (Op.intShiftRight bc lhs rhs) s = (do
      let x ← evalOpR R lhs s
      let y ← evalOpR R rhs s
      pure ⟨fun i => x.data (bc.leftIndex i) >>> y.data (bc.rightIndex i)⟩) := by
  cases hx : evalOpR R lhs s <;> cases hy : evalOpR R rhs s <;>
    simp only [Op.intShiftRight, evalOpR, hx, hy, bind, pure,
      Option.bind_none, Option.bind_some, Tile.bop, Tile.scalar,
      IntegralDType.floorDiv, IntegerExpr.same_left, IntegerExpr.same_right]
  congr 1
  ext i
  simp only [Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, one_mul,
    Int.ofNat_eq_natCast, Nat.cast_pow, Nat.cast_ofNat]

theorem evalOpR_intBitAnd (R : RoundingModel) (width : Nat) (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .int b) (s : BlockState) :
    evalOpR R (Op.intBitAnd width bc lhs rhs) s = (do
      let x ← evalOpR R lhs s
      let y ← evalOpR R rhs s
      pure ⟨fun i => ((BitVec.ofInt width (x.data (bc.leftIndex i))) &&&
        (BitVec.ofInt width (y.data (bc.rightIndex i)))).toInt⟩) := by
  rw [Op.intBitAnd, evalOpR_unsignedToInt]
  simp only [evalOpR, evalOpR_intToUnsigned]
  cases hx : evalOpR R lhs s <;> cases hy : evalOpR R rhs s <;>
    simp only [hx, hy, bind, pure, Option.bind_none, Option.bind_some]
  congr 1
  ext i
  dsimp only
  rw [signedBits_eq_bitvec]
  congr 1
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat, BitVec.toNat_and, BitVec.toNat_ofInt,
    Tile.bop, Nat.cast_pow, Nat.cast_ofNat]
  apply Nat.mod_eq_of_lt
  exact lt_of_le_of_lt (Nat.and_le_left) (unsigned_cast_lt width _)

theorem FP.Structural.eval_unsignedToInt {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (width : Nat)
    (value : Op .nat shape) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.unsignedToInt width value) s = (do
      let t ← FP.Structural.evalOp M p value s
      pure (fun i => IntegerExpr.signedBits width (t i))) := by
  cases h : FP.Structural.evalOp M p value s <;>
    simp only [Op.unsignedToInt, Op.natToUnsigned, FP.Structural.evalOp, h,
      bind, pure, Option.bind_none, Option.bind_some, FP.Structural.numeric,
      FP.Structural.bop, FP.Structural.integralMod, FP.Structural.numericLt,
      FP.Structural.natLt, IntegerExpr.same_left, IntegerExpr.same_right,
      IntegerExpr.signedBits, decide_eq_true_eq]
  rfl

theorem FP.Structural.eval_intShiftRight {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (bc : Broadcast a b out)
    (lhs : Op .int a) (rhs : Op .nat b) (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.intShiftRight bc lhs rhs) s = (do
      let x ← FP.Structural.evalOp M p lhs s
      let y ← FP.Structural.evalOp M p rhs s
      pure (fun i => x (bc.leftIndex i) >>> y (bc.rightIndex i))) := by
  cases hx : FP.Structural.evalOp M p lhs s <;>
    cases hy : FP.Structural.evalOp M p rhs s <;>
    simp only [Op.intShiftRight, FP.Structural.evalOp, hx, hy, bind, pure,
      Option.bind_none, Option.bind_some, FP.Structural.bop,
      FP.Structural.integralFloorDiv, IntegerExpr.same_left, IntegerExpr.same_right]
  apply congrArg some
  funext i
  simp only [Int.shiftRight_eq_div_pow, Nat.shiftLeft_eq, one_mul,
    Int.ofNat_eq_natCast, Nat.cast_pow, Nat.cast_ofNat, FP.Structural.bop]

theorem FP.Structural.eval_intBitAnd {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (p : Option ComputeDType) (width : Nat)
    (bc : Broadcast a b out) (lhs : Op .int a) (rhs : Op .int b)
    (s : FP.Structural.State α) :
    FP.Structural.evalOp M p (Op.intBitAnd width bc lhs rhs) s = (do
      let x ← FP.Structural.evalOp M p lhs s
      let y ← FP.Structural.evalOp M p rhs s
      pure (fun i => ((BitVec.ofInt width (x (bc.leftIndex i))) &&&
        (BitVec.ofInt width (y (bc.rightIndex i)))).toInt)) := by
  rw [Op.intBitAnd, FP.Structural.eval_unsignedToInt]
  simp only [FP.Structural.evalOp, FP.Structural.eval_intToUnsigned]
  cases hx : FP.Structural.evalOp M p lhs s <;>
    cases hy : FP.Structural.evalOp M p rhs s <;>
    simp only [hx, hy, bind, pure, Option.bind_none, Option.bind_some]
  apply congrArg some
  funext i
  rw [signedBits_eq_bitvec]
  congr 1
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_ofNat, BitVec.toNat_and, BitVec.toNat_ofInt,
    FP.Structural.bop, FP.Structural.integralMod, IntegerExpr.same_left,
    Nat.cast_pow, Nat.cast_ofNat]
  apply Nat.mod_eq_of_lt
  exact lt_of_le_of_lt (Nat.and_le_left) (unsigned_cast_lt width _)

end VeriTile.Triton

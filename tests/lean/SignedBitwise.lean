import VeriTile.Triton.Float.FixedInteger
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Tests.SignedBitwise
open VeriTile.Triton

def low (offset : Int) : Op .nat [] :=
  Op.intToUnsigned 32 (Op.intBitAnd 64 .nil (.constInt offset) (.constInt 4294967295))

def high (offset : Int) : Op .nat [] :=
  Op.intToUnsigned 32 (Op.intBitAnd 64 .nil
    (Op.intShiftRight .nil (.constInt offset) (.constNat 32)) (.constInt 4294967295))

theorem negative_offset (s : BlockState) :
    evalOp (low (-1)) s = some (Tile.scalar 4294967295) ∧
    evalOp (high (-1)) s = some (Tile.scalar 4294967295) := by
  simp only [low, high, eval_intToUnsigned, eval_intBitAnd, eval_intShiftRight,
    evalOp, bind, pure, Option.bind_some, Tile.scalar]
  constructor <;> rfl

theorem minimum_offset (s : BlockState) :
    evalOp (low (-(2^63))) s = some (Tile.scalar 0) ∧
    evalOp (high (-(2^63))) s = some (Tile.scalar 2147483648) := by
  simp only [low, high, eval_intToUnsigned, eval_intBitAnd, eval_intShiftRight,
    evalOp, bind, pure, Option.bind_some, Tile.scalar]
  constructor <;> rfl

theorem empty_tile_still_evaluates (s : BlockState)
    (h : s.regs .int [0] "missing" = none) :
    evalOp (Op.intBitAnd 64 (IntegerExpr.sameShape [0])
      (.ref .int [0] "missing") (.full [0] (.constInt (-1)))) s = none := by
  simp only [eval_intBitAnd, evalOp, h, bind, Option.bind_none]

#axiomsClean VeriTile.Triton.signedBits_eq_bitvec
#axiomsClean VeriTile.Triton.eval_unsignedToInt
#axiomsClean VeriTile.Triton.eval_intBitAnd
#axiomsClean VeriTile.Triton.eval_intShiftRight
#axiomsClean VeriTile.Triton.evalOpR_unsignedToInt
#axiomsClean VeriTile.Triton.evalOpR_intBitAnd
#axiomsClean VeriTile.Triton.evalOpR_intShiftRight
#axiomsClean VeriTile.Triton.FP.Structural.eval_unsignedToInt
#axiomsClean VeriTile.Triton.FP.Structural.eval_intBitAnd
#axiomsClean VeriTile.Triton.FP.Structural.eval_intShiftRight
#axiomsClean negative_offset
#axiomsClean minimum_offset
#axiomsClean empty_tile_still_evaluates
end VeriTile.Bench.Tests.SignedBitwise

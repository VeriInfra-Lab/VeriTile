import bench.tritonbench_g.uniform_sampling.Proofs.Mathematical_ScalarSpec
import VeriTile.Triton.Float.FixedInteger

/-! Small bit-vector bridges for the exact source counter expressions.
These local mathematical identities do not waive the original source seed
and offset signed64 ABI premises required by the complete kernel contract.
-/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.CounterOperands
open VeriTile.Triton ScalarSpec
noncomputable section

theorem truncate_ofInt {small large : Nat} (h : small ≤ large) (value : Int) :
    (BitVec.ofInt large value).setWidth small = BitVec.ofInt small value := by
  cases value with
  | ofNat n =>
      change (BitVec.ofInt large (n : Int)).setWidth small = BitVec.ofInt small (n : Int)
      simp only [BitVec.ofInt_natCast, BitVec.setWidth_ofNat_of_le h]
  | negSucc n =>
      simp only [BitVec.ofInt_negSucc_eq_not_ofNat, BitVec.setWidth_not h,
        BitVec.setWidth_ofNat_of_le h]

theorem truncate_signed {small large : Nat} (h : small ≤ large) (value : BitVec large) :
    BitVec.ofInt small value.toInt = value.setWidth small := by
  have he := congrArg (BitVec.setWidth small) (BitVec.ofInt_toInt (x := value))
  rw [truncate_ofInt h] at he
  exact he

/-- Signed AND at width64 and the subsequent uint32 cast retain precisely
the low word, including for negative original offsets. -/
theorem masked_low32 (value : Int) :
    BitVec.ofInt 32
      (((BitVec.ofInt 64 value) &&& BitVec.ofNat 64 0xffffffff).toInt) =
      BitVec.ofInt 32 value := by
  rw [truncate_signed (by decide : 32 ≤ 64), BitVec.setWidth_and,
    truncate_ofInt (by decide : 32 ≤ 64)]
  have hm : (BitVec.ofNat 64 0xffffffff).setWidth 32 = BitVec.allOnes 32 := by
    decide +kernel
  rw [hm, BitVec.and_allOnes]

theorem eval_low_mask (value : Int) (valueOp : Op .int []) (s : BlockState)
    (hv : evalOp valueOp s = some (Tile.scalar value)) :
    evalOp (Op.intToUnsigned 32 (Op.intBitAnd 64 .nil valueOp (.constInt 0xffffffff))) s =
      some (Tile.scalar (BitVec.ofInt 32 value).toNat) := by
  rw [eval_intToUnsigned, eval_intBitAnd, hv]
  simp only [evalOp.eq_def, bind, Option.bind, pure, Tile.scalar]
  change some (Tile.scalar (dtype := .nat)
    (((((BitVec.ofInt 64 value) &&& BitVec.ofInt 64 0xffffffff).toInt) % (2^32 : Int)).toNat)) = _
  rw [unsigned_cast_eq_bitvec]
  change some (Tile.scalar (dtype := .nat)
    (BitVec.ofInt 32 (((BitVec.ofInt 64 value) &&& BitVec.ofNat 64 0xffffffff).toInt)).toNat) = _
  rw [masked_low32]
  rfl

theorem eval_high_mask (offset : Int) (offsetOp : Op .int []) (s : BlockState)
    (ho : evalOp offsetOp s = some (Tile.scalar offset)) :
    evalOp (Op.intToUnsigned 32 (Op.intBitAnd 64 .nil
      (Op.intShiftRight .nil offsetOp (.constNat 32)) (.constInt 0xffffffff))) s =
      some (Tile.scalar (counterHigh offset).toNat) := by
  have hs : evalOp (Op.intShiftRight .nil offsetOp (.constNat 32)) s =
      some (Tile.scalar (offset >>> 32)) := by
    rw [eval_intShiftRight, ho]
    simp only [evalOp.eq_def, bind, Option.bind, pure, Tile.scalar]
  rw [eval_low_mask _ _ _ hs, Int.shiftRight_eq_div_pow]
  rfl

/-- The source first materializes the unsigned low word, then promotes the
signed lane index to uint32 for addition. The result wraps even when the
unbounded mathematical sum exceeds 2^32; no carry changes the high word. -/
theorem wrapped_low_add (offset : Int) (i4 : Nat) :
    ((((BitVec.ofInt 32 offset).toNat : Int) + (i4 : Int)) % (2^32 : Int)).toNat =
      (counterLow offset i4).toNat := by
  rw [unsigned_cast_eq_bitvec]
  simp only [BitVec.ofInt_add, BitVec.ofInt_natCast, BitVec.ofNat_toNat,
    BitVec.setWidth_eq, counterLow]

#axiomsClean truncate_ofInt
#axiomsClean truncate_signed
#axiomsClean masked_low32
#axiomsClean eval_low_mask
#axiomsClean eval_high_mask
#axiomsClean wrapped_low_add

end
end VeriTile.Bench.TritonBenchG.UniformSampling.CounterOperands

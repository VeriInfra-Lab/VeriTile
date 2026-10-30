import bench.tritonbench_g.matmul_dequant_int4.Proofs.Binary16
import VeriTile.Triton.Float.FixedInteger
import VeriTile.Triton.Float.ScalarOps
import Mathlib.Tactic
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits
open VeriTile.Triton
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

/-- The true int32 arithmetic shift and mask agree with unsigned packed-nibble
extraction for all eight positions, including negative packed words. -/
theorem signed_logical_mask (word : BitVec 32) (slot : Fin 8) :
    (word.sshiftRight (slot.val * 4) &&& (15 : BitVec 32)) =
      ((word >>> (slot.val * 4)) &&& (15 : BitVec 32)) := by
  apply BitVec.eq_of_getElem_eq
  intro i hi
  by_cases hlow : i < 4
  · have hsum : slot.val * 4 + i < 32 := by omega
    simp [BitVec.getElem_sshiftRight, BitVec.getElem_ushiftRight, BitVec.getLsbD_eq_getElem, hsum]
  · have hmask : (15 : Nat).testBit i = false := by
      interval_cases i <;> decide +kernel
    simp [BitVec.getElem_and, BitVec.getElem_eq_testBit_toNat, BitVec.toNat_ofNat, hmask]

theorem mask_bound (word : BitVec 32) :
    0 ≤ (word &&& (15 : BitVec 32)).toInt ∧ (word &&& (15 : BitVec 32)).toInt ≤ 15 := by
  have hn : (word &&& (15 : BitVec 32)).toNat ≤ 15 := by
    simpa only [BitVec.toNat_and, BitVec.toNat_ofNat, Nat.reduceMod] using
      (Nat.and_le_right (n := word.toNat) (m := 15))
  rw [BitVec.toInt_eq_toNat_of_lt (by omega)]
  omega

theorem difference_no_overflow (a b : BitVec 32) :
    (BitVec.ofInt 32 ((a &&& (15 : BitVec 32)).toInt - (b &&& (15 : BitVec 32)).toInt)).toInt =
      (a &&& (15 : BitVec 32)).toInt - (b &&& (15 : BitVec 32)).toInt := by
  have ha := mask_bound a
  have hb := mask_bound b
  apply BitVec.toInt_ofInt_eq_self (by decide)
  · omega
  · omega

/-- All possible signed nibble differences have an exact fp16 conversion.
This is a finite theorem about the concrete software format, not an assumed
identity cast for arbitrary real inputs. -/
theorem fp16_difference_exact (n : Int) (hn : -15 ≤ n ∧ n ≤ 15) :
    (Binary16.round .ieeeValue (n : Rat)).toRat? = some (n : Rat) := by
  rcases hn with ⟨hl, hu⟩
  interval_cases n <;> decide +kernel

#axiomsClean signed_logical_mask
#axiomsClean mask_bound
#axiomsClean difference_no_overflow
#axiomsClean fp16_difference_exact
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4.FPBits

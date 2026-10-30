import VeriTile.Triton.Float.ScalarOps
import VeriTile.Triton.Random.Proofs.RoundDyadic
import VeriTile.Meta.StatementAudit

/-! Strict COUNT-ZERO and COUNT-SUCCESSOR for the concrete FP32 software model.
The profile is nearest-even with neither input nor result flushing. The proof
reuses the shared dyadic rounding theorem; no empirical admission, hardware
conformance premise, or assumed count-conversion equation is used here. -/
namespace VeriTile.Triton.FP.CountConversion.Exact

open VeriTile.Triton.Random

/-- All integers through this endpoint are exactly representable in FP32.
The successor rule requires its input to be strictly below the endpoint. -/
def limit : Nat := 2^24

/-- Mathematical nonnegative count converted by the actual FP32 rounder. -/
def fromNat (i : Nat) : Value .fp32 := round .fp32 .ieeeValue (i : Rat)

/-- The selected signed-int32 conversion, including its signed interpretation. -/
def fromInt32 (i : BitVec 32) : Value .fp32 :=
  round .fp32 .ieeeValue (Rat.ofInt i.toInt)

/-- The first 24-bit rounding step is exact throughout the count range.
Only the single endpoint uses a coarser rounding unit; the interior is proved
using the logarithmic significand bound, not by enumerating the integers. -/
theorem round24_exact (i : Nat) (hi : i ≤ limit) : Dyadic.round24 i = i := by
  rcases lt_or_eq_of_le hi with hi | rfl
  · by_cases hz : i = 0
    · subst i
      decide +kernel
    · have hlog : i.log2 < 24 := (Nat.log2_lt hz).mpr hi
      have hstep : i.log2 - 23 = 0 := by omega
      simp [Dyadic.round24, Dyadic.unit24, hstep, Dyadic.roundMultiple,
        Dyadic.roundQuotient]
      omega
  · decide +kernel

/-- Exact finite decoding of every count needed by a recurrence of length
`N ≤ 2^24`, including the last count `2^24`. -/
theorem fromNat_decode (i : Nat) (hi : i ≤ limit) :
    (fromNat i).decode = .finite false (i : Rat) := by
  have hlarge : i < 2^55 := lt_of_le_of_lt hi (by norm_num [limit])
  have h := RoundBridge.round_dyadic_decode i 0 hlarge (by decide)
  simpa [fromNat, round24_exact i hi] using h

/-- Exact addition for two positive-sign finite operands in this profile.
The result is still rounded, so this statement makes no general exact-addition
or reassociation claim. -/
theorem add_of_nonnegative_decodes (x y : Value .fp32) (a b : Rat)
    (hx : x.decode = .finite false a) (hy : y.decode = .finite false b) :
    FP.add .ieeeValue x y = round .fp32 .ieeeValue (a+b) := by
  unfold FP.add prepare
  simp only [Config.ieeeValue, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  rw [hx, hy]
  change round .fp32 .ieeeValue (a+b)
    (if a == 0 && b == 0 && (false == false) then false else false) = _
  simp [Config.ieeeValue]

/-- COUNT-ZERO is an exact theorem of the selected conversion model. -/
theorem count_zero : fromNat 0 = Value.zero .fp32 := by
  rfl

/-- The increment is the usual exact FP32 literal one. -/
theorem fromNat_one : fromNat 1 = Value.ofBits .fp32 0x3f800000 := by
  decide +kernel

/-- COUNT-SUCCESSOR retains the strict input bound. Both sides execute the
actual FP32 rounder, and the right side executes the actual FP32 addition. -/
theorem count_successor (i : Nat) (hi : i < limit) :
    fromNat (i+1) = FP.add .ieeeValue (fromNat i) (fromNat 1) := by
  rw [add_of_nonnegative_decodes _ _ _ _
    (fromNat_decode i (Nat.le_of_lt hi)) (fromNat_decode 1 (by norm_num [limit]))]
  simp [fromNat]

/-- Exact facts for every step of a bounded recurrence. The bound is retained
on `N`; it is not replaced by an unbounded conversion interface. -/
theorem count_steps (N : Nat) (hN : N ≤ limit) :
    fromNat 0 = Value.zero .fp32 ∧
      ∀ i, i < N → fromNat (i+1) = FP.add .ieeeValue (fromNat i) (fromNat 1) :=
  ⟨count_zero, fun i hi => count_successor i (lt_of_lt_of_le hi hN)⟩

/-- Every count, including the endpoint, has its intended signed-int32 value. -/
theorem int32_ofNat_toInt (i : Nat) (hi : i ≤ limit) :
    (BitVec.ofNat 32 i).toInt = (i : Int) := by
  rw [BitVec.toInt_ofNat']
  apply Int.bmod_eq_of_le
  · norm_num
  · norm_num [limit] at hi ⊢
    omega

/-- The natural-count and signed-int32 conversion presentations coincide on
the full recurrence range. -/
theorem fromInt32_ofNat (i : Nat) (hi : i ≤ limit) :
    fromInt32 (BitVec.ofNat 32 i) = fromNat i := by
  unfold fromInt32
  rw [int32_ofNat_toInt i hi]
  rfl

/-- Reinterpret any nonnegative signed-int32 input as its mathematical count. -/
theorem fromInt32_eq_fromNat (i : BitVec 32) (hi : 0 ≤ i.toInt) :
    fromInt32 i = fromNat i.toInt.toNat := by
  change round .fp32 .ieeeValue (i.toInt : Rat) =
    round .fp32 .ieeeValue (i.toInt.toNat : Rat)
  congr 1
  exact_mod_cast (Int.toNat_of_nonneg hi).symm

/-- Incrementing an in-range count in signed int32 does not overflow. -/
theorem int32_successor_toInt (i : BitVec 32)
    (hi0 : 0 ≤ i.toInt) (hi : i.toInt < (limit : Int)) :
    (i + 1).toInt = i.toInt + 1 := by
  rw [BitVec.toInt_add]
  change (i.toInt + 1).bmod (2^32) = _
  apply Int.bmod_eq_of_le <;> norm_num [limit] at hi ⊢ <;> omega

/-- COUNT-ZERO for the selected signed-int32 conversion. -/
theorem count_int32_zero : fromInt32 0 = Value.zero .fp32 := by
  rfl

/-- COUNT-SUCCESSOR with an actual signed-int32 increment on the left and an
actual FP32 addition on the right. The input bound is strict. -/
theorem count_int32_successor (i : BitVec 32)
    (hi0 : 0 ≤ i.toInt) (hi : i.toInt < (limit : Int)) :
    fromInt32 (i + 1) = FP.add .ieeeValue (fromInt32 i) (fromNat 1) := by
  have hsucc := int32_successor_toInt i hi0 hi
  have hsucc0 : 0 ≤ (i + 1).toInt := by omega
  rw [fromInt32_eq_fromNat (i + 1) hsucc0, fromInt32_eq_fromNat i hi0]
  have hnat : (i + 1).toInt.toNat = i.toInt.toNat + 1 := by omega
  rw [hnat]
  exact count_successor i.toInt.toNat (by omega)

#axiomsClean round24_exact
#axiomsClean fromNat_decode
#axiomsClean add_of_nonnegative_decodes
#axiomsClean count_zero
#axiomsClean fromNat_one
#axiomsClean count_successor
#axiomsClean count_steps
#axiomsClean int32_ofNat_toInt
#axiomsClean fromInt32_ofNat
#axiomsClean fromInt32_eq_fromNat
#axiomsClean int32_successor_toInt
#axiomsClean count_int32_zero
#axiomsClean count_int32_successor

end VeriTile.Triton.FP.CountConversion.Exact

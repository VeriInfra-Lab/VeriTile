import VeriTile.Triton.Float.BitValueLaws
import VeriTile.Triton.Float.CastExact.Packed

/-! Classification of results of exact rational rounding in the concrete
software model. Finite rational inputs may overflow, but cannot create NaNs. -/
namespace VeriTile.Triton.FP.BitValueRoundClassification

open VeriTile.Triton.Random.RoundBridge
open BitValueLaws

private theorem nonnegative_num_div_den (q : Rat) (hq : 0 ≤ q) :
    (q.num.natAbs : Rat) / q.den = q := by
  have hn : (q.num.natAbs : Int) = q.num := by
    rw [Int.natCast_natAbs, abs_of_nonneg (Rat.num_nonneg.mpr hq)]
  rw [← Int.cast_natCast, hn, Rat.num_div_den]

/-- The numerator/denominator logarithm estimate is at most one above the
true binary exponent, even for a non-dyadic rational. -/
theorem estimate_upper (q : Rat) (hq : 0 ≤ q) :
    q < FP.pow2 ((q.num.natAbs.log2 : Int) - q.den.log2 + 1) := by
  have hd : (2 : Rat)^q.den.log2 ≤ q.den := by
    exact_mod_cast Nat.log2_self_le q.den_nz
  have hn : (q.num.natAbs : Rat) < (2 : Rat)^(q.num.natAbs.log2+1) := by
    exact_mod_cast Nat.lt_log2_self (n := q.num.natAbs)
  calc
    q = (q.num.natAbs : Rat) / q.den := (nonnegative_num_div_den q hq).symm
    _ ≤ (q.num.natAbs : Rat) / (2 : Rat)^q.den.log2 :=
      div_le_div_of_nonneg_left (by positivity) (by positivity) hd
    _ < (2 : Rat)^(q.num.natAbs.log2+1) / (2 : Rat)^q.den.log2 :=
      div_lt_div_of_pos_right hn (by positivity)
    _ = FP.pow2 ((q.num.natAbs.log2 : Int) - q.den.log2 + 1) := by
      rw [pow2_eq_zpow]
      rw [show (q.num.natAbs.log2 : Int) - q.den.log2 + 1 =
        ((q.num.natAbs.log2+1 : Nat) : Int) - (q.den.log2 : Int) by omega]
      rw [zpow_sub₀ (by norm_num : (2 : Rat) ≠ 0), zpow_natCast, zpow_natCast]

/-- The format-dependent scale used by the existing `FP.round`. -/
def roundStep (f : Format) (q : Rat) : Int :=
  let estimate : Int := (q.num.natAbs.log2 : Int) - q.den.log2
  let exponent := if q < FP.pow2 estimate then estimate-1 else estimate
  max exponent (-126) - f.fractionBits

theorem roundStep_lower (f : Format) (q : Rat) :
    -126 ≤ roundStep f q + f.fractionBits := by
  dsimp only [roundStep]
  have := le_max_right (if q < FP.pow2 ((q.num.natAbs.log2 : Int)-q.den.log2)
    then (q.num.natAbs.log2 : Int)-q.den.log2-1
    else (q.num.natAbs.log2 : Int)-q.den.log2) (-126)
  omega

theorem scaled_upper (f : Format) (q : Rat) (hq : 0 ≤ q) :
    q / FP.pow2 (roundStep f q) < (2 : Rat)^(f.fractionBits+1) := by
  let estimate : Int := (q.num.natAbs.log2 : Int)-q.den.log2
  let exponent := if q < FP.pow2 estimate then estimate-1 else estimate
  have he : q < FP.pow2 (exponent+1) := by
    dsimp only [exponent]
    split
    · simpa only [sub_add_cancel] using ‹q < FP.pow2 estimate›
    · exact estimate_upper q hq
  have hm : FP.pow2 (exponent+1) ≤
      FP.pow2 (roundStep f q + ((f.fractionBits+1 : Nat) : Int)) := by
    rw [pow2_eq_zpow, pow2_eq_zpow]
    apply zpow_le_zpow_right₀ (by norm_num : (1 : Rat) ≤ 2)
    change exponent+1 ≤ max exponent (-126)-f.fractionBits+(f.fractionBits+1 : Nat)
    have := le_max_left exponent (-126)
    push_cast
    omega
  apply (div_lt_iff₀ (pow2_pos _)).mpr
  have hp : (2 : Rat)^(f.fractionBits+1) * FP.pow2 (roundStep f q) =
      FP.pow2 (roundStep f q + ((f.fractionBits+1 : Nat) : Int)) := by
    rw [pow2_eq_zpow, pow2_eq_zpow,
      zpow_add₀ (by norm_num : (2 : Rat) ≠ 0), zpow_natCast, mul_comm]
  rw [hp]
  exact lt_of_lt_of_le he hm

/-- Rounding upward by at most one cannot cross an integer upper bound. -/
theorem roundRat_le (q : Rat) (n : Nat) (hq : 0 ≤ q) (h : q < n) :
    roundRat q ≤ n := by
  have hnum : q.num.natAbs < n*q.den := by
    have hr : (q.num.natAbs : Rat) < (n : Rat)*q.den := by
      apply (div_lt_iff₀ (by exact_mod_cast q.den_pos : (0 : Rat) < q.den)).mp
      rwa [nonnegative_num_div_den q hq]
    exact_mod_cast hr
  have hdiv : q.num.natAbs / q.den < n :=
    (Nat.div_lt_iff_lt_mul q.den_pos).mpr hnum
  exact le_trans (VeriTile.Triton.Random.Dyadic.roundQuotient_le_succ _ _) (by omega)

theorem significand_upper (f : Format) (q : Rat) (hq : 0 ≤ q) :
    roundRat (q / FP.pow2 (roundStep f q)) ≤ 2^(f.fractionBits+1) := by
  apply roundRat_le _ _ (div_nonneg hq (pow2_pos _).le)
  simpa only [Nat.cast_pow, Nat.cast_ofNat] using scaled_upper f q hq

private def packRounded (f : Format) (negative : Prop) [Decidable negative]
    (step : Int) (n : Nat) : Value f :=
  let carry := n ≥ 2^(f.fractionBits+1)
  let significand := if carry then n/2 else n
  let exponent := step+f.fractionBits+(if carry then 1 else 0)
  if exponent > 127 then Value.infinity f (decide negative)
  else if significand < 2^f.fractionBits then
    Value.ofBits f ((if negative then 2^(f.fractionBits+8) else 0)+significand)
  else Value.ofBits f ((if negative then 2^(f.fractionBits+8) else 0)+
    (exponent+127).toNat*2^f.fractionBits+significand-2^f.fractionBits)

private theorem packRounded_not_nan (f : Format) (negative : Prop) [Decidable negative]
    (step : Int) (n : Nat) (hstep : -126 ≤ step+f.fractionBits)
    (hn : n ≤ 2^(f.fractionBits+1)) :
    (packRounded f negative step n).isNaN = false := by
  let significand := if n ≥ 2^(f.fractionBits+1) then n/2 else n
  let exponent := step+f.fractionBits+(if n ≥ 2^(f.fractionBits+1) then 1 else 0)
  have hs : significand < 2^(f.fractionBits+1) := by
    dsimp only [significand]
    split
    · cases f <;> norm_num [Format.fractionBits] at * <;> omega
    · omega
  have he : -126 ≤ exponent := by
    dsimp only [exponent]
    split <;> omega
  change (if exponent > 127 then Value.infinity f (decide negative)
    else if significand < 2^f.fractionBits then
      Value.ofBits f ((if negative then 2^(f.fractionBits+8) else 0)+significand)
    else Value.ofBits f ((if negative then 2^(f.fractionBits+8) else 0)+
      (exponent+127).toNat*2^f.fractionBits+significand-2^f.fractionBits)).isNaN = false
  split
  · cases f <;> cases (decide negative) <;> rfl
  · split
    · simpa only [decide_eq_true_eq] using
        CastExact.subnormal_pack_not_nan f (decide negative) significand ‹_›
    · simpa only [decide_eq_true_eq] using
        CastExact.normal_pack_not_nan f (decide negative) exponent significand
          ⟨he, by omega⟩ ⟨by omega, hs⟩

private def referenceRound (f : Format) (q : Rat) (negativeZero : Bool) : Value f :=
  if q == 0 then Value.zero f negativeZero
  else packRounded f (q < 0) (roundStep f q.abs)
    (roundRat (q.abs / FP.pow2 (roundStep f q.abs)))

private theorem round_reference (f : Format) (q : Rat) (negativeZero : Bool) :
    round f .ieeeValue q negativeZero = referenceRound f q negativeZero := by
  unfold referenceRound
  rw [roundRat_formula]
  unfold round packRounded roundStep
  rfl

/-- The concrete nearest-even rounding operation cannot turn a rational into
a NaN. Overflow produces infinity and preserves its distinction from NaN. -/
theorem round_not_nan (f : Format) (q : Rat) (negativeZero : Bool) :
    (round f .ieeeValue q negativeZero).isNaN = false := by
  rw [round_reference]
  unfold referenceRound
  split
  · cases f <;> cases negativeZero <;> rfl
  · exact packRounded_not_nan f (q < 0) _ _ (roundStep_lower f q.abs)
      (significand_upper f q.abs Rat.abs_nonneg)

#axiomsClean estimate_upper
#axiomsClean significand_upper
#axiomsClean round_not_nan

end VeriTile.Triton.FP.BitValueRoundClassification

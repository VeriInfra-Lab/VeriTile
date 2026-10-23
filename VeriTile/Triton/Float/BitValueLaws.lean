import VeriTile.Triton.Random.Proofs.DyadicStep
import VeriTile.Triton.Float.CastExact.Fields
import VeriTile.Meta.StatementAudit

/-! Exact representable dyadic rounding for the existing concrete FP model.
The logarithm/normalization facts are reused from the shared random conversion
proofs. These are model theorems, not empirical or hardware-conformance claims. -/
namespace VeriTile.Triton.FP.BitValueLaws

open VeriTile.Triton.Random.RoundBridge

theorem pow2_pos (e : Int) : (0 : Rat) < FP.pow2 e := by
  rw [pow2_eq_zpow]
  positivity

theorem pow2_ne_zero (e : Int) : FP.pow2 e ≠ 0 := ne_of_gt (pow2_pos e)

/-- Exact exponent estimate of a positive dyadic; no denominator-range limit. -/
theorem scaled_exponent_estimate (n : Nat) (e : Int) (hn : 0 < n) :
    (((n : Rat) * FP.pow2 e).num.natAbs.log2 : Int) -
      ((n : Rat) * FP.pow2 e).den.log2 = (n.log2 : Int) + e := by
  cases e with
  | ofNat k =>
    simp only [Int.ofNat_eq_natCast]
    have hq : (n : Rat) * FP.pow2 (k : Int) = ((2^k*n : Nat) : Rat) := by
      rw [pow2_eq_zpow, zpow_natCast]
      push_cast
      ring
    rw [hq]
    simp only [Rat.num_natCast, Rat.den_natCast, Int.natAbs_natCast,
      show Nat.log2 1 = 0 from rfl,
      Nat.cast_zero, sub_zero, log2_pow_mul n k (Nat.ne_of_gt hn), Nat.cast_add]
    omega
  | negSucc k =>
    have hq : (n : Rat) * FP.pow2 (Int.negSucc k) = (n : Rat) / 2^(k+1) := by
      rw [pow2_eq_zpow]
      change (n : Rat) * (2 : Rat)^(-((k+1 : Nat) : Int)) = _
      rw [zpow_neg, zpow_natCast, div_eq_mul_inv]
    rw [hq, dyadic_exponent_estimate n (k+1) hn]
    omega

theorem scaled_lower (n : Nat) (e : Int) (hn : 0 < n) :
    FP.pow2 ((n.log2 : Int) + e) ≤ (n : Rat) * FP.pow2 e := by
  rw [pow2_eq_zpow, zpow_add₀ (by norm_num : (2 : Rat) ≠ 0), zpow_natCast,
    ← pow2_eq_zpow]
  apply mul_le_mul_of_nonneg_right _ (pow2_pos e).le
  exact_mod_cast Nat.log2_self_le (Nat.ne_of_gt hn)

/-- A nonzero significand which already fits its chosen normal/subnormal scale. -/
def ExactScale (f : Format) (n : Nat) (e : Int) : Prop :=
  0 < n ∧ n < 2^(f.fractionBits+1) ∧
    -126-(f.fractionBits : Int) ≤ e ∧ e+(f.fractionBits : Int) ≤ 127 ∧
    (2^f.fractionBits ≤ n ∨ e = -126-(f.fractionBits : Int))

theorem exact_scale_step (f : Format) (n : Nat) (e : Int)
    (h : ExactScale f n e) :
    max ((n.log2 : Int) + e) (-126) - f.fractionBits = e := by
  rcases h with ⟨hn, hbound, hmin, hmax, hnormal | hsub⟩
  · have hlo : f.fractionBits ≤ n.log2 :=
      (Nat.le_log2 (Nat.ne_of_gt hn)).mpr hnormal
    have hhi : n.log2 < f.fractionBits + 1 :=
      (Nat.log2_lt (Nat.ne_of_gt hn)).mpr hbound
    have he : n.log2 = f.fractionBits := by omega
    rw [he, max_eq_left (by omega)]
    omega
  · have hhi : n.log2 < f.fractionBits + 1 :=
      (Nat.log2_lt (Nat.ne_of_gt hn)).mpr hbound
    rw [max_eq_right (by omega)]
    omega

/-- The explicit packed result of an exactly representable nonzero dyadic. -/
def packed (f : Format) (negative : Bool) (n : Nat) (e : Int) : Value f :=
  Value.ofBits f (if n < 2^f.fractionBits then
    (if negative then 2^(f.fractionBits+8) else 0) + n
  else (if negative then 2^(f.fractionBits+8) else 0) +
    (e+f.fractionBits+127).toNat * 2^f.fractionBits + n - 2^f.fractionBits)

private def roundStep (f : Format) (m : Rat) : Int :=
  let estimate : Int := (m.num.natAbs.log2 : Int) - m.den.log2
  let exponent := if m < FP.pow2 estimate then estimate-1 else estimate
  max exponent (-126) - f.fractionBits

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

private theorem roundRat_nat (n : Nat) : roundRat (n : Rat) = n := by
  have h := roundRat_nat_div n 1 (by decide)
  norm_num [VeriTile.Triton.Random.Dyadic.roundQuotient] at h
  rw [Nat.mod_one] at h
  norm_num at h
  exact h

/-- Rounding an already representable dyadic changes no bit. The explicit
scale bounds include the entire subnormal range of both supported formats. -/
theorem round_scaled (f : Format) (negative : Bool) (n : Nat) (e : Int)
    (h : ExactScale f n e) :
    round f .ieeeValue (if negative then -((n : Rat)*FP.pow2 e)
      else (n : Rat)*FP.pow2 e) negative = packed f negative n e := by
  have hm : (0 : Rat) < (n : Rat)*FP.pow2 e :=
    mul_pos (by exact_mod_cast h.1) (pow2_pos e)
  have hstep : roundStep f ((n : Rat)*FP.pow2 e) = e := by
    unfold roundStep
    rw [scaled_exponent_estimate n e h.1]
    dsimp only
    rw [if_neg (not_lt_of_ge (scaled_lower n e h.1))]
    exact exact_scale_step f n e h
  have hs : roundRat (((n : Rat)*FP.pow2 e)/FP.pow2 e) = n := by
    rw [mul_div_cancel_right₀ _ (pow2_ne_zero e), roundRat_nat]
  rw [round_reference]
  cases negative <;>
    simp only [referenceRound, Bool.false_eq_true, ↓reduceIte,
      beq_iff_eq, ne_of_gt hm, neg_ne_zero.mpr (ne_of_gt hm), Rat.abs_neg,
      Rat.abs_of_nonneg hm.le, hstep, hs]
  all_goals
    unfold packRounded packed
    simp only [not_lt_of_ge hm.le, neg_lt_zero.mpr hm, ↓reduceIte,
      show ¬ 2^(f.fractionBits+1) ≤ n from not_le.mpr h.2.1,
      Int.add_zero, show ¬e+(f.fractionBits : Int)>127 from not_lt.mpr h.2.2.2.1,
      Bool.false_eq_true]
    split <;> rfl

/-- Exact finite decode/round inverse. The decoded sign is passed separately,
so the theorem includes both signed zeros without identifying their words. -/
theorem round_of_decode_finite {f : Format} (x : Value f) (negative : Bool) (m : Rat)
    (h : x.decode = .finite negative m) :
    round f .ieeeValue (if negative then -m else m) negative = x := by
  have he255 : x.exponent ≠ 255 := by
    intro he
    simp only [Value.decode, he, beq_self_eq_true, ↓reduceIte] at h
    split at h <;> cases h
  have hfrac := CastExact.fraction_lt x
  have hexp := CastExact.exponent_lt x
  by_cases he : x.exponent = 0
  · have hd : Decoded.finite x.sign
        ((x.fraction : Rat)*FP.pow2 (-126-(f.fractionBits : Int))) =
        .finite negative m := by
      simpa [Value.decode, he] using h
    cases hd
    by_cases hz : x.fraction = 0
    · have hx : Value.zero f x.sign = x := by
        change Value.ofBits f ((if x.sign then 2^(f.fractionBits+8) else 0)+0*2^f.fractionBits+0)=x
        simpa only [he,hz] using CastExact.ofBits_fields x
      simpa [hz,round] using hx
    · have hs : ExactScale f x.fraction (-126-(f.fractionBits : Int)) := by
        refine ⟨by omega, ?_, le_refl _, by omega, Or.inr rfl⟩
        cases f <;> norm_num [Format.fractionBits] at hfrac ⊢ <;> omega
      rw [round_scaled f x.sign x.fraction _ hs]
      unfold packed
      rw [if_pos hfrac]
      simpa only [he, Nat.zero_mul, Nat.add_zero] using CastExact.ofBits_fields x
  · have hd : Decoded.finite x.sign
        (((2^f.fractionBits+x.fraction : Nat) : Rat)*
          FP.pow2 ((x.exponent : Int)-127-f.fractionBits)) =
        .finite negative m := by
      simpa [Value.decode, he255, he] using h
    cases hd
    have hs : ExactScale f (2^f.fractionBits+x.fraction)
        ((x.exponent : Int)-127-f.fractionBits) := by
      refine ⟨by positivity, ?_, by omega, by omega, Or.inl (by omega)⟩
      cases f <;> norm_num [Format.fractionBits] at hfrac ⊢ <;> omega
    rw [round_scaled f x.sign _ _ hs]
    unfold packed
    rw [if_neg (by omega)]
    have hexact : (x.exponent : Int)-127-f.fractionBits+f.fractionBits+127 =
        (x.exponent : Int) := by omega
    rw [hexact, Int.toNat_natCast]
    convert CastExact.ofBits_fields x using 1
    congr 1
    omega

#axiomsClean pow2_pos
#axiomsClean scaled_exponent_estimate
#axiomsClean exact_scale_step
#axiomsClean round_scaled
#axiomsClean round_of_decode_finite

end VeriTile.Triton.FP.BitValueLaws

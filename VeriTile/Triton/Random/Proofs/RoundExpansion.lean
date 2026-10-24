import VeriTile.Triton.Random.Proofs.RoundQuotientBridge

namespace VeriTile.Triton.Random.RoundBridge
open Dyadic

theorem roundRat_formula (q : Rat) :
    roundRat q =
      let n := q.num.natAbs / q.den
      let rem := q.num.natAbs % q.den
      if 2*rem > q.den || (2*rem == q.den && n%2 == 1) then n+1 else n := by
  unfold roundRat roundQuotient
  simp only [Bool.or_eq_true, decide_eq_true_eq, Bool.and_eq_true, beq_iff_eq]
  by_cases h : q.den < 2*(q.num.natAbs % q.den) ∨
      (q.den = 2*(q.num.natAbs % q.den) ∧ q.num.natAbs / q.den % 2 = 1)
  · have h' : q.den < 2*(q.num.natAbs % q.den) ∨
      (2*(q.num.natAbs % q.den) = q.den ∧ q.num.natAbs / q.den % 2 = 1) := by tauto
    simp only [h, h', ↓reduceIte]
  · have h' : ¬(q.den < 2*(q.num.natAbs % q.den) ∨
      (2*(q.num.natAbs % q.den) = q.den ∧ q.num.natAbs / q.den % 2 = 1)) := by tauto
    simp only [h, h', ↓reduceIte, Nat.add_zero]

/-- The scale exponent selected by canonical rounding. -/
def roundStep (magnitude : Rat) : Int :=
  let estimate : Int := (magnitude.num.natAbs.log2 : Int) - magnitude.den.log2
  let exponent := if magnitude < FP.pow2 estimate then estimate-1 else estimate
  max exponent (-126) - 23

/-- The canonical carry and packing computation, exposed as a small proof
interface so rewriting the rounding step does not duplicate its whole term. -/
def packRounded (negative : Prop) [Decidable negative] (step : Int) (n : Nat) : FP.Value .fp32 :=
  let carry := n ≥ 2^24
  let significand := if carry then n/2 else n
  let exponent := step+23+(if carry then 1 else 0)
  if exponent > 127 then FP.Value.infinity .fp32 (decide negative)
  else if significand < 2^23 then
    FP.Value.ofBits .fp32 ((if negative then 2^31 else 0) + significand)
  else FP.Value.ofBits .fp32
    ((if negative then 2^31 else 0) + (exponent+127).toNat * 2^23 + significand - 2^23)

/-- A factored proof interface to the existing operation; the generator continues
using FP.round itself. -/
def referenceRound (q : Rat) (negativeZero : Bool := false) : FP.Value .fp32 :=
  if q == 0 then FP.Value.zero .fp32 negativeZero
  else packRounded (q < 0) (roundStep q.abs)
    (roundRat (q.abs / FP.pow2 (roundStep q.abs)))

set_option maxRecDepth 2000 in
theorem round_eq_referenceRound (q : Rat) (negativeZero : Bool := false) :
    FP.round .fp32 FP.Config.ieeeValue q negativeZero = referenceRound q negativeZero := by
  unfold referenceRound
  rw [roundRat_formula]
  unfold FP.round packRounded roundStep
  rfl

/-- Nonnegative arguments have no hidden sign or magnitude conversion. -/
theorem round_nonnegative (q : Rat) (hq : 0 ≤ q) :
    FP.round .fp32 FP.Config.ieeeValue q =
      if q = 0 then FP.Value.zero .fp32
      else packRounded False (roundStep q) (roundRat (q / FP.pow2 (roundStep q))) := by
  rw [round_eq_referenceRound]
  unfold referenceRound
  simp only [beq_iff_eq, Rat.abs_of_nonneg hq]
  by_cases hz : q = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    unfold packRounded
    simp only [not_lt_of_ge hq, decide_false, ↓reduceIte]

end VeriTile.Triton.Random.RoundBridge

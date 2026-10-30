import VeriTile.Triton.Random.Philox
import VeriTile.Triton.Random.Proofs.Dyadic
import Mathlib.Tactic

namespace VeriTile.Triton.Random.RoundBridge
open Dyadic

/-- The ties-even rational quotient used inside the canonical FP.round. -/
def roundRat (q : Rat) : Nat := roundQuotient q.num.natAbs q.den

theorem roundQuotient_scale (n d c : Nat) (hc : 0 < c) :
    roundQuotient (c*n) (c*d) = roundQuotient n d := by
  have hlt : c*d < 2*(c*(n%d)) ↔ d < 2*(n%d) := by
    rw [show 2*(c*(n%d)) = c*(2*(n%d)) by ring]
    exact mul_lt_mul_iff_right₀ hc
  have heq : c*d = 2*(c*(n%d)) ↔ d = 2*(n%d) := by
    rw [show 2*(c*(n%d)) = c*(2*(n%d)) by ring]
    constructor
    · exact Nat.mul_left_cancel hc
    · intro h; exact congrArg (fun z => c*z) h
  simp only [roundQuotient, Nat.mul_div_mul_left _ _ hc,
    Nat.mul_mod_mul_left, hlt, heq]

/-- Rational normalization cannot change the nearest-even quotient. -/
theorem roundRat_nat_div (n d : Nat) (hd : 0 < d) :
    roundRat ((n:Rat)/d) = roundQuotient n d := by
  let q : Rat := (n:Rat)/d
  have hqn : 0 ≤ q.num := Rat.num_nonneg.mpr (by dsimp [q]; positivity)
  have hqd : 0 < (q.den:Int) := by exact_mod_cast q.den_pos
  obtain ⟨c, hn, hden⟩ := Rat.exists_eq_mul_div_num_and_eq_mul_div_den
    (n:Int) (show (d:Int) ≠ 0 by exact_mod_cast Nat.ne_of_gt hd)
  change (n:Int) = c*q.num at hn
  change (d:Int) = c*q.den at hden
  have hc : 0 < c := by
    have hd' : 0 < (d:Int) := by exact_mod_cast hd
    nlinarith
  have hcn : (c.toNat:Int) = c := Int.toNat_of_nonneg (by omega)
  have hcpos : 0 < c.toNat := by omega
  have hnum : (q.num.natAbs:Int) = q.num := by
    rw [Int.natCast_natAbs, abs_of_nonneg hqn]
  have hn' : n = c.toNat * q.num.natAbs := by exact_mod_cast (by nlinarith : (n:Int) = (c.toNat:Int)*(q.num.natAbs:Int))
  have hd' : d = c.toNat * q.den := by exact_mod_cast (by nlinarith : (d:Int) = (c.toNat:Int)*(q.den:Int))
  change roundQuotient q.num.natAbs q.den = _
  conv_rhs => rw [hn', hd']
  exact (roundQuotient_scale _ _ _ hcpos).symm

end VeriTile.Triton.Random.RoundBridge

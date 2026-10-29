import VeriTile.Triton.Random.Proofs.RoundExpansion
import VeriTile.Triton.Random.Proofs.DyadicExponent

namespace VeriTile.Triton.Random.RoundBridge

theorem pow2_eq_zpow (e : Int) : FP.pow2 e = (2:Rat)^e := by
  cases e with
  | ofNat n =>
      simp only [FP.pow2, Int.ofNat_eq_natCast,
        Int.toNat_natCast, zpow_natCast]
      change (((2:Nat)^n : Nat) : Rat) = _
      norm_cast
  | negSucc n =>
      rw [FP.pow2, if_pos (by omega)]
      change mkRat 1 (2^(n+1)) = (2:Rat)^(-((n+1:Nat):Int))
      rw [zpow_neg, zpow_natCast, Rat.mkRat_eq_divInt, Rat.divInt_eq_div]
      simp only [Int.cast_one, Nat.cast_pow, Nat.cast_ofNat, one_div]
      norm_cast

theorem pow2_sub_nat (a b : Nat) :
    FP.pow2 ((a:Int)-b) = (2:Rat)^a / (2:Rat)^b := by
  rw [pow2_eq_zpow, zpow_sub₀ (by norm_num : (2:Rat) ≠ 0),
    zpow_natCast, zpow_natCast]

/-- Positive dyadics in this denominator range do not take the subnormal clamp
or the one-step estimate correction. -/
theorem dyadic_roundStep (n k : Nat) (hn : 0 < n) (hk : k ≤ 126) :
    roundStep ((n:Rat)/2^k) = (n.log2:Int)-k-23 := by
  unfold roundStep
  rw [dyadic_exponent_estimate n k hn]
  have hlow : FP.pow2 ((n.log2:Int)-k) ≤ (n:Rat)/2^k := by
    rw [pow2_sub_nat]
    apply (div_le_div_iff_of_pos_right (by positivity : (0:Rat)<2^k)).mpr
    exact_mod_cast Nat.log2_self_le (Nat.ne_of_gt hn)
  dsimp only
  rw [if_neg (not_lt_of_ge hlow), max_eq_left (by omega)]

/-- Power-of-two scaling cancels before rounding the significand. -/
theorem dyadic_significand_argument (n k : Nat) :
    ((n:Rat)/2^k) / FP.pow2 ((n.log2:Int)-k-23) =
      (n:Rat) / FP.pow2 ((n.log2:Int)-23) := by
  have hexp : (n.log2:Int)-k-23 = ((n.log2:Int)-23) - k := by omega
  rw [hexp, pow2_eq_zpow, pow2_eq_zpow,
    zpow_sub₀ (by norm_num : (2:Rat) ≠ 0), zpow_natCast]
  field_simp

end VeriTile.Triton.Random.RoundBridge

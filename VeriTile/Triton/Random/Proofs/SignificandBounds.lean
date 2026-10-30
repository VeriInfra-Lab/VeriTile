import VeriTile.Triton.Random.Proofs.DyadicStep

namespace VeriTile.Triton.Random.RoundBridge
open Dyadic

set_option maxRecDepth 2000
set_option maxHeartbeats 2000000

theorem roundRat_nat (n : Nat) : roundRat (n:Rat) = n := by
  simpa [roundQuotient, Nat.mod_one] using roundRat_nat_div n 1 (by decide)

/-- The canonical normalized significand is in the carry-safe interval, and
its exact scale product agrees with integer 24-bit rounding. -/
theorem normalized_significand (n : Nat) (hn : 0 < n) :
    let s := roundRat ((n:Rat) / FP.pow2 ((n.log2:Int)-23))
    2^23 ≤ s ∧ s ≤ 2^24 ∧
      (s:Rat)*FP.pow2 ((n.log2:Int)-23) = (round24 n:Rat) := by
  have hlo := Nat.log2_self_le (Nat.ne_of_gt hn)
  have hhi := Nat.lt_log2_self (n := n)
  dsimp only
  by_cases hl : n.log2 ≤ 23
  · have he : (n.log2:Int)-23 = -((23-n.log2:Nat):Int) := by omega
    have hp : FP.pow2 ((n.log2:Int)-23) = 1/(2:Rat)^(23-n.log2) := by
      rw [he, pow2_eq_zpow, zpow_neg, zpow_natCast, one_div]
    have harg : (n:Rat)/FP.pow2 ((n.log2:Int)-23) =
        ((n*2^(23-n.log2):Nat):Rat) := by
      rw [hp]
      simp only [one_div, div_inv_eq_mul, Nat.cast_mul, Nat.cast_pow, Nat.cast_ofNat]
    rw [harg, roundRat_nat]
    have hlo' : 2^23 ≤ n*2^(23-n.log2) := by
      calc
        2^23 = 2^n.log2 * 2^(23-n.log2) := by rw [← pow_add];congr 1;omega
        _ ≤ n*2^(23-n.log2) := Nat.mul_le_mul_right _ hlo
    have hhi' : n*2^(23-n.log2) < 2^24 := by
      calc
        n*2^(23-n.log2) < 2^(n.log2+1)*2^(23-n.log2) :=
          Nat.mul_lt_mul_of_pos_right hhi (by positivity)
        _ = 2^24 := by rw [← pow_add];congr 1;omega
    refine ⟨hlo', by omega, ?_⟩
    have hr : round24 n = n := by
      simp [round24, roundMultiple, unit24, Nat.sub_eq_zero_of_le hl, roundQuotient, Nat.mod_one]
    rw [hp,hr]
    push_cast
    field_simp
  · have hl' : 23 ≤ n.log2 := by omega
    let d : Nat := 2^(n.log2-23)
    have hd : 0 < d := by dsimp [d];positivity
    have he : (n.log2:Int)-23 = ((n.log2-23:Nat):Int) := by omega
    have hp : FP.pow2 ((n.log2:Int)-23) = (d:Rat) := by
      rw [he,pow2_eq_zpow,zpow_natCast]
      dsimp [d]
      norm_cast
    rw [hp,roundRat_nat_div n d hd]
    have hlo' : 2^23*d ≤ n := by
      have heq : 2^23*d = 2^n.log2 := by change 2^23 * 2^(n.log2-23) = 2^n.log2;rw [←pow_add];congr 1;omega
      rwa [heq]
    have hhi' : n < 2^24*d := by
      have heq : 2^24*d = 2^(n.log2+1) := by change 2^24 * 2^(n.log2-23) = 2^(n.log2+1);rw [←pow_add];congr 1;omega
      rwa [heq]
    have hqlo : 2^23 ≤ n/d := (Nat.le_div_iff_mul_le hd).mpr hlo'
    have hqhi : n/d < 2^24 := (Nat.div_lt_iff_lt_mul hd).mpr hhi'
    have hrl : n/d ≤ roundQuotient n d := by unfold roundQuotient;omega
    refine ⟨by omega, le_trans (roundQuotient_le_succ n d) (by omega), ?_⟩
    change (roundQuotient n d:Rat)*(d:Rat) = ((roundQuotient n d*d:Nat):Rat)
    push_cast
    rfl

end VeriTile.Triton.Random.RoundBridge

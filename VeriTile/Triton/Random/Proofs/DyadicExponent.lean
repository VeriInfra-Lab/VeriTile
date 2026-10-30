import VeriTile.Triton.Random.Proofs.RoundQuotientBridge

namespace VeriTile.Triton.Random.RoundBridge

theorem normalization_factors (n d : Nat) (hd : 0 < d) :
    ∃ c : Nat, 0 < c ∧ n = c * (((n:Rat)/d).num.natAbs) ∧
      d = c * ((n:Rat)/d).den := by
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
  have hnum : (q.num.natAbs:Int) = q.num := by
    rw [Int.natCast_natAbs, abs_of_nonneg hqn]
  refine ⟨c.toNat, by omega, ?_, ?_⟩
  · exact_mod_cast (by nlinarith : (n:Int) = (c.toNat:Int)*(q.num.natAbs:Int))
  · exact_mod_cast (by nlinarith : (d:Int) = (c.toNat:Int)*(q.den:Int))

theorem log2_pow_mul (n k : Nat) (hn : n ≠ 0) :
    (2^k*n).log2 = k+n.log2 := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [pow_succ', Nat.mul_assoc, Nat.log2_two_mul (by positivity), ih]
    omega

/-- Reducing a dyadic rational cancels the same power of two from numerator and
denominator, so the exact log2 exponent estimate is invariant. -/
theorem dyadic_exponent_estimate (n k : Nat) (hn : 0 < n) :
    (((n:Rat)/2^k).num.natAbs.log2 : Int) - ((n:Rat)/2^k).den.log2 =
      (n.log2:Int) - k := by
  let q : Rat := (n:Rat)/2^k
  obtain ⟨c,hc,hnc,hdc⟩ := normalization_factors n (2^k) (by positivity)
  simp only [Nat.cast_pow, Nat.cast_ofNat] at hnc hdc
  change n = c*q.num.natAbs at hnc
  change 2^k = c*q.den at hdc
  have hcdiv : c ∣ 2^k := ⟨q.den,hdc⟩
  obtain ⟨s,_,hcs⟩ := (Nat.dvd_prime_pow Nat.prime_two).mp hcdiv
  have hnum : q.num.natAbs ≠ 0 := by intro hz; rw [hz,Nat.mul_zero] at hnc; omega
  have hlogn := congrArg Nat.log2 hnc
  have hlogd := congrArg Nat.log2 hdc
  rw [hcs, log2_pow_mul _ _ hnum] at hlogn
  rw [Nat.log2_two_pow, hcs, log2_pow_mul _ _ q.den_nz] at hlogd
  change (q.num.natAbs.log2:Int) - q.den.log2 = _
  omega

end VeriTile.Triton.Random.RoundBridge

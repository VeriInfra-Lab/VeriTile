import Mathlib.Tactic

/-!
Auxiliary exact positive dyadic rounding lemmas. These do not replace the
canonical `Random.uniform` defined via the existing FP.round and FP.mul.
Random.Proofs.CanonicalUniformBridge proves that software implementation
equal to the `uniformDyadic` expression below for every input word.
-/
namespace VeriTile.Triton.Random.Dyadic

/-- Nearest-integer quotient, choosing an even quotient on an exact half tie. -/
def roundQuotient (n d : Nat) : Nat :=
  n / d + if d < 2 * (n % d) ∨ (d = 2 * (n % d) ∧ n / d % 2 = 1) then 1 else 0

/-- Round a nonnegative integer to the nearest multiple of a positive unit. -/
def roundMultiple (n d : Nat) : Nat := roundQuotient n d * d

theorem roundQuotient_le_succ (n d : Nat) : roundQuotient n d ≤ n / d + 1 := by
  unfold roundQuotient
  split <;> omega

theorem roundQuotient_tie_even (n d : Nat) (htie : d = 2 * (n % d)) :
    roundQuotient n d % 2 = 0 := by
  have hm := Nat.mod_lt (n / d) (by decide : 0 < 2)
  unfold roundQuotient
  split <;> omega

theorem roundMultiple_le_multiple (n d k : Nat) (hd : 0 < d) (hn : n ≤ k * d) :
    roundMultiple n d ≤ k * d := by
  have hq : n / d ≤ k := Nat.div_le_of_le_mul (by simpa [Nat.mul_comm] using hn)
  by_cases hlt : n / d < k
  · exact Nat.mul_le_mul_right d (le_trans (roundQuotient_le_succ n d) (by omega))
  · have heq : n / d = k := by omega
    have hid := Nat.div_add_mod n d
    have hr : n % d = 0 := by
      rw [heq, Nat.mul_comm d k] at hid
      omega
    simp [roundMultiple, roundQuotient, heq, hr, Nat.ne_of_gt hd]

theorem roundMultiple_le_of_dvd (n d bound : Nat) (hd : 0 < d)
    (hn : n ≤ bound) (hdiv : d ∣ bound) : roundMultiple n d ≤ bound := by
  rcases hdiv with ⟨k, hk⟩
  subst bound
  simpa [Nat.mul_comm] using roundMultiple_le_multiple n d k hd (by simpa [Nat.mul_comm] using hn)

/-- For a positive integer significand, fp32 retains 24 binary significant
bits. The zero branch uses unit one and remains exactly zero. -/
def unit24 (n : Nat) : Nat := 2 ^ (n.log2 - 23)

def round24 (n : Nat) : Nat := roundMultiple n (unit24 n)

theorem unit24_pos (n : Nat) : 0 < unit24 n := by unfold unit24; positivity

theorem unit24_dvd_pow (n k : Nat) (hn : n < 2 ^ (k+24)) :
    unit24 n ∣ 2 ^ k := by
  apply Nat.pow_dvd_pow
  by_cases hz : n = 0
  · simp [hz]
  · have hl : n.log2 < k+24 := (Nat.log2_lt hz).2 hn
    omega

theorem first_round_bound (n : Nat) (hn : n < 2^31) : round24 n ≤ 2^31 := by
  apply roundMultiple_le_of_dvd n (unit24 n) (2^31) (unit24_pos n) (by omega)
  apply unit24_dvd_pow n 31
  exact lt_trans hn (by norm_num)

/-- Exact numerator of the second dyadic multiplication, before its rounding. -/
def productNumerator (n : Nat) : Nat := round24 n * (2^24 - 1)

/-- A closed positive-dyadic reference, proved equal to FP.round/FP.mul in the uniform bridge.
The denominator comes from the exact fp32 scale 0x2fffffff. -/
def uniformDyadic (n : Nat) : ℚ := (round24 (productNumerator n) : ℚ) / 2^55

theorem productNumerator_bound (n : Nat) (hn : n < 2^31) :
    productNumerator n ≤ 2^31 * (2^24-1) := by
  exact Nat.mul_le_mul_right _ (first_round_bound n hn)

theorem second_round_bound (n : Nat) (hn : n < 2^31) :
    round24 (productNumerator n) ≤ 2^31 * (2^24-1) := by
  have hnum := productNumerator_bound n hn
  have hsmall : productNumerator n < 2^55 := lt_of_le_of_lt hnum (by norm_num)
  apply roundMultiple_le_of_dvd _ _ _ (unit24_pos _) hnum
  exact dvd_mul_of_dvd_left (unit24_dvd_pow (productNumerator n) 31 hsmall) _

theorem uniformDyadic_range (n : Nat) (hn : n < 2^31) :
    0 ≤ uniformDyadic n ∧ uniformDyadic n ≤ 1 - (1:ℚ)/2^24 ∧ uniformDyadic n < 1 := by
  have hb : (round24 (productNumerator n) : ℚ) ≤ (2^31 * (2^24-1) : ℚ) := by
    have hb0 := second_round_bound n hn
    norm_num at hb0 ⊢
    exact_mod_cast hb0
  unfold uniformDyadic
  constructor
  · positivity
  constructor
  · apply (div_le_iff₀ (by positivity)).2
    convert hb using 1
    norm_num
  · apply (div_lt_one (by positivity)).2
    exact lt_of_le_of_lt hb (by norm_num)

end VeriTile.Triton.Random.Dyadic

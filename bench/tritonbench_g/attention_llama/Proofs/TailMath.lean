import VeriTile.Triton.Math.Attention

/-! The normalized accumulator used by attention_llama, including masked
partial blocks. The independent target is an exponential weighted sum over
the logical keys. No execution result occurs in the invariant's assumptions.
The selected finite maximum is arbitrary: its value cancels algebraically. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.TailMath
open scoped BigOperators
set_option maxHeartbeats 800000

structure State where
  maximum : ℝ
  denominator : ℝ
  accumulator : ℝ

noncomputable def mass (score : Nat → ℝ) (keys : Finset Nat) : ℝ :=
  ∑ k ∈ keys, Real.exp (score k)

noncomputable def moment (score value : Nat → ℝ) (keys : Finset Nat) : ℝ :=
  ∑ k ∈ keys, Real.exp (score k) * value k

noncomputable def expected (score value : Nat → ℝ) (keys : Finset Nat) : ℝ :=
  moment score value keys / mass score keys

noncomputable def step (score value : Nat → ℝ) (keys : Finset Nat)
    (selectedMaximum : ℝ) (state : State) : State :=
  let previous := state.denominator * Real.exp (state.maximum - selectedMaximum)
  let probability := fun k => Real.exp (score k - selectedMaximum)
  let current := (∑ k ∈ keys, probability k) + previous
  let reciprocal := 1 / current
  { maximum := selectedMaximum
    denominator := current
    accumulator := state.accumulator * (previous * reciprocal) +
      ∑ k ∈ keys, (probability k * reciprocal) * value k }

def Consistent (score value : Nat → ℝ) (keys : Finset Nat) (state : State) : Prop :=
  state.denominator = Real.exp (-state.maximum) * mass score keys ∧
  state.accumulator = expected score value keys

theorem mass_nonneg (score : Nat → ℝ) (keys : Finset Nat) :
    0 ≤ mass score keys := by
  exact Finset.sum_nonneg (fun _ _ => le_of_lt (Real.exp_pos _))

theorem mass_pos (score : Nat → ℝ) {keys : Finset Nat} (h : keys.Nonempty) :
    0 < mass score keys := by
  exact Finset.sum_pos (fun _ _ => Real.exp_pos _) h

theorem mass_eq_zero_iff (score : Nat → ℝ) (keys : Finset Nat) :
    mass score keys = 0 ↔ keys = ∅ := by
  constructor
  · intro h
    by_contra hn
    exact (ne_of_gt (mass_pos score (Finset.nonempty_iff_ne_empty.mpr hn))) h
  · rintro rfl
    simp [mass]

theorem scaled_mass (score : Nat → ℝ) (keys : Finset Nat) (m : ℝ) :
    (∑ k ∈ keys, Real.exp (score k - m)) = Real.exp (-m) * mass score keys := by
  simp only [mass, Finset.mul_sum, sub_eq_add_neg, Real.exp_add]
  apply Finset.sum_congr rfl
  intro k _
  ring

theorem scaled_moment (score value : Nat → ℝ) (keys : Finset Nat) (m r : ℝ) :
    (∑ k ∈ keys, (Real.exp (score k - m) * r) * value k) =
      Real.exp (-m) * r * moment score value keys := by
  simp only [moment, Finset.mul_sum, sub_eq_add_neg, Real.exp_add]
  apply Finset.sum_congr rfl
  intro k _
  ring

theorem shift_cancel (a b : ℝ) :
    Real.exp (-a) * Real.exp (a - b) = Real.exp (-b) := by
  rw [← Real.exp_add]
  congr 1
  ring

theorem step_consistent (score value : Nat → ℝ) (seen keys : Finset Nat)
    (m : ℝ) (state : State) (hd : Disjoint seen keys)
    (h : Consistent score value seen state) :
    Consistent score value (seen ∪ keys) (step score value keys m state) := by
  obtain ⟨hl, ha⟩ := h
  have hmass : mass score (seen ∪ keys) = mass score seen + mass score keys := by
    exact Finset.sum_union hd
  have hmoment : moment score value (seen ∪ keys) =
      moment score value seen + moment score value keys := by
    exact Finset.sum_union hd
  have hprev : state.denominator * Real.exp (state.maximum - m) =
      Real.exp (-m) * mass score seen := by
    rw [hl]
    calc
      Real.exp (-state.maximum) * mass score seen * Real.exp (state.maximum - m) =
          (Real.exp (-state.maximum) * Real.exp (state.maximum - m)) * mass score seen := by ring
      _ = _ := by rw [shift_cancel]
  have hcurrent : (∑ k ∈ keys, Real.exp (score k - m)) +
      state.denominator * Real.exp (state.maximum - m) =
      Real.exp (-m) * (mass score seen + mass score keys) := by
    rw [scaled_mass, hprev]
    ring
  constructor
  · change (∑ k ∈ keys, Real.exp (score k - m)) +
        state.denominator * Real.exp (state.maximum - m) = _
    rw [hcurrent, hmass]
    rfl
  · change state.accumulator *
        (state.denominator * Real.exp (state.maximum - m) *
          (1 / ((∑ k ∈ keys, Real.exp (score k - m)) +
            state.denominator * Real.exp (state.maximum - m)))) +
        (∑ k ∈ keys, (Real.exp (score k - m) *
          (1 / ((∑ j ∈ keys, Real.exp (score j - m)) +
            state.denominator * Real.exp (state.maximum - m)))) * value k) = _
    rw [hcurrent, hprev, ha, scaled_moment]
    simp only [expected, hmass, hmoment]
    by_cases hz : mass score seen + mass score keys = 0
    · simp [hz]
    · have he := ne_of_gt (Real.exp_pos (-m))
      by_cases hs : mass score seen = 0
      · have hempty := (mass_eq_zero_iff score seen).mp hs
        subst seen
        simp only [mass, moment, Finset.sum_empty, zero_div, zero_mul, zero_add] at *
        field_simp
      · field_simp

noncomputable def initial : State := ⟨0, 0, 0⟩

theorem initial_consistent (score value : Nat → ℝ) :
    Consistent score value ∅ initial := by
  simp [Consistent, initial, mass, expected, moment]

noncomputable def run (score value : Nat → ℝ) (select : Nat → ℝ)
    (blocks : Nat → Finset Nat) : Nat → State
  | 0 => initial
  | n + 1 => step score value (blocks n) (select n) (run score value select blocks n)

def processed (blocks : Nat → Finset Nat) : Nat → Finset Nat
  | 0 => ∅
  | n + 1 => processed blocks n ∪ blocks n

theorem run_consistent (score value : Nat → ℝ) (select : Nat → ℝ)
    (blocks : Nat → Finset Nat) (n : Nat)
    (hd : ∀ t < n, Disjoint (processed blocks t) (blocks t)) :
    Consistent score value (processed blocks n) (run score value select blocks n) := by
  induction n with
  | zero => exact initial_consistent score value
  | succ n ih =>
    exact step_consistent score value _ _ _ _ (hd n (Nat.lt_succ_self n))
      (ih (fun t ht => hd t (Nat.lt_trans ht (Nat.lt_succ_self n))))

theorem run_eq_weighted_sum (score value : Nat → ℝ) (select : Nat → ℝ)
    (blocks : Nat → Finset Nat) (n : Nat)
    (hd : ∀ t < n, Disjoint (processed blocks t) (blocks t)) :
    (run score value select blocks n).accumulator =
      expected score value (processed blocks n) :=
  (run_consistent score value select blocks n hd).2

end VeriTile.Bench.Repairs.AttentionLlama.TailMath

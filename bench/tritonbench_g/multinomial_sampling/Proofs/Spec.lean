import VeriTile.Triton.Random.Values
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-! Input-only categorical inverse-CDF specification. The canonical Philox
sample is the exact Real decoding of the source's two-rounded FP32 uniform,
not an ideal continuously distributed random variable. -/
namespace VeriTile.Bench.Repairs.MultinomialSampling.Spec
open VeriTile.Triton
open scoped BigOperators
noncomputable section

def cumulative (weights : Nat → ℝ) (n : Nat) : ℝ := ∑ j ∈ Finset.range n, weights j

def Interval (weights : Nat → ℝ) (u : ℝ) (j : Nat) : Prop :=
  cumulative weights j < u ∧ u ≤ cumulative weights (j + 1)

def Position (tokens : Nat) (weights : Nat → ℝ) (u : ℝ) (j : Nat) : Prop :=
  j < tokens ∧ ((u = 0 ∧ j = 0) ∨ (0 < u ∧ Interval weights u j))

def Domain (tokens : Nat) (weights : Nat → ℝ) : Prop :=
  0 < tokens ∧ (∀ j, j < tokens → 0 ≤ weights j) ∧ cumulative weights tokens = 1

def sample (seed offset : Int) : ℝ :=
  Random.randReal seed (BitVec.ofInt 32 offset).toInt .bits32 10

/-- The mathematical output specification is relational so no arbitrary
choice or fallback program is hidden in its definition. Uniqueness below
makes it a deterministic label for each supplied canonical sample. -/
def Result (tokens : Nat) (weights : Nat → ℝ) (labels : Nat → Int)
    (seed offset output : Int) : Prop :=
  ∃ j, Position tokens weights (sample seed offset) j ∧ output = labels j

theorem prefix_zero (weights : Nat → ℝ) : cumulative weights 0 = 0 := by
  simp [cumulative]

theorem prefix_succ (weights : Nat → ℝ) (n : Nat) :
    cumulative weights (n+1) = cumulative weights n + weights n := by
  simp [cumulative, Finset.sum_range_succ]

theorem prefix_mono (tokens : Nat) (weights : Nat → ℝ)
    (nonnegative : ∀ j, j < tokens → 0 ≤ weights j)
    (a b : Nat) (ab : a ≤ b) (bt : b ≤ tokens) :
    cumulative weights a ≤ cumulative weights b := by
  induction b, ab using Nat.le_induction with
  | base => exact le_rfl
  | succ b ab ih =>
    rw [prefix_succ]
    exact le_trans (ih (by omega)) (le_add_of_nonneg_right (nonnegative b (by omega)))

theorem interval_unique (tokens : Nat) (weights : Nat → ℝ)
    (nonnegative : ∀ j, j < tokens → 0 ≤ weights j)
    (u : ℝ) (i j : Nat) (it : i < tokens) (jt : j < tokens)
    (hi : Interval weights u i) (hj : Interval weights u j) : i = j := by
  rcases lt_trichotomy i j with h | h | h
  · have hm := prefix_mono tokens weights nonnegative (i+1) j (by omega) (by omega)
    exact False.elim (not_lt_of_ge (le_trans hi.2 hm) hj.1)
  · exact h
  · have hm := prefix_mono tokens weights nonnegative (j+1) i (by omega) (by omega)
    exact False.elim (not_lt_of_ge (le_trans hj.2 hm) hi.1)

theorem position_unique (tokens : Nat) (weights : Nat → ℝ)
    (domain : Domain tokens weights) (u : ℝ) (i j : Nat)
    (hi : Position tokens weights u i) (hj : Position tokens weights u j) : i = j := by
  rcases hi with ⟨it, hi | hi⟩ <;> rcases hj with ⟨jt, hj | hj⟩
  · omega
  · exact False.elim (by linarith [hi.1, hj.1])
  · exact False.elim (by linarith [hi.1, hj.1])
  · exact interval_unique tokens weights domain.2.1 u i j it jt hi.2 hj.2

theorem interval_exists (tokens : Nat) (weights : Nat → ℝ)
    (positive : 0 < tokens) (u : ℝ) (up : 0 < u) (ut : u ≤ cumulative weights tokens) :
    ∃ j, j < tokens ∧ Interval weights u j := by
  induction tokens with
  | zero => omega
  | succ n ih =>
    by_cases before : u ≤ cumulative weights n
    · have np : 0 < n := by
        by_contra h
        have nz : n = 0 := by omega
        subst n
        rw [prefix_zero] at before
        linarith
      obtain ⟨j, hj, hit⟩ := ih np before
      exact ⟨j, by omega, hit⟩
    · exact ⟨n, by omega, lt_of_not_ge before, ut⟩

theorem position_exists (tokens : Nat) (weights : Nat → ℝ)
    (domain : Domain tokens weights) (u : ℝ) (range : 0 ≤ u ∧ u < 1) :
    ∃ j, Position tokens weights u j := by
  by_cases zero : u = 0
  · exact ⟨0, domain.1, Or.inl ⟨zero, rfl⟩⟩
  · have up : 0 < u := lt_of_le_of_ne range.1 (Ne.symm zero)
    obtain ⟨j, jt, hj⟩ := interval_exists tokens weights domain.1 u up (by rw [domain.2.2]; exact le_of_lt range.2)
    exact ⟨j, jt, Or.inr ⟨up, hj⟩⟩

theorem sample_range (seed offset : Int) : 0 ≤ sample seed offset ∧ sample seed offset < 1 := by
  have h := Random.randReal_range seed (BitVec.ofInt 32 offset).toInt .bits32 10
  exact ⟨h.1, h.2.2⟩

theorem result_exists_unique (tokens : Nat) (weights : Nat → ℝ) (labels : Nat → Int)
    (domain : Domain tokens weights) (seed offset : Int) :
    ∃! output, Result tokens weights labels seed offset output := by
  obtain ⟨j, hj⟩ := position_exists tokens weights domain _ (sample_range seed offset)
  refine ⟨labels j, ⟨j, hj, rfl⟩, ?_⟩
  rintro out ⟨k, hk, ho⟩
  rw [ho, position_unique tokens weights domain _ k j hk hj]

theorem singleton (weights : Nat → ℝ) (labels : Nat → Int)
    (domain : Domain 1 weights) (seed offset : Int) :
    Result 1 weights labels seed offset (labels 0) := by
  obtain ⟨j, hj⟩ := position_exists 1 weights domain _ (sample_range seed offset)
  have jzero : j = 0 := by have := hj.1; omega
  exact ⟨j, hj, by rw [jzero]⟩

#axiomsClean result_exists_unique
#axiomsClean singleton
#specNonCircular Result avoiding [FP.Structural.exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MultinomialSampling.Spec

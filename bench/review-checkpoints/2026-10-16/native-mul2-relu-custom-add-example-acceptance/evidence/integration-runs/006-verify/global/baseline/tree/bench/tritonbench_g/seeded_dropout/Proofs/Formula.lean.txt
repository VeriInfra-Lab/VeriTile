import VeriTile.Triton.Random.DropoutFormula
import Mathlib.Tactic

/-! The specification uses the fixed canonical generator, runtime seed and p.
The pure formula's real division is total; its p=1 result does not permit
removing the eagerly evaluated floating division in the source program. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Formula
open VeriTile.Triton
noncomputable section

def keep (seed : Int) (p : ℝ) (offset : Nat) : Bool :=
  decide (Random.randReal seed (offset : Int) .bits32 10 > p)

def expected (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat) : ℝ :=
  Random.seededDropoutExpected input p seed offset

theorem expected_eq_if (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat) :
    expected input p seed offset =
      if Random.randReal seed (offset : Int) .bits32 10 > p
      then input offset / (1-p) else 0 := rfl

theorem expected_congr_input (input other : Nat → ℝ) (p : ℝ)
    (seed : Int) (offset : Nat) (h : input offset = other offset) :
    expected input p seed offset = expected other p seed offset := by
  simp only [expected_eq_if, h]

theorem expected_keep (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat)
    (h : keep seed p offset = true) :
    expected input p seed offset = input offset / (1-p) := by
  simp only [keep, decide_eq_true_eq] at h
  simp only [expected_eq_if, h, if_true]

theorem expected_drop (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat)
    (h : keep seed p offset = false) : expected input p seed offset = 0 := by
  simp only [keep, decide_eq_false_iff_not] at h
  simp only [expected_eq_if, h, if_false]

theorem expected_zero_input (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat)
    (h : input offset = 0) : expected input p seed offset = 0 := by
  simp only [expected_eq_if, h, zero_div, ite_self]

theorem expected_of_one_le (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat)
    (hp : 1 ≤ p) : expected input p seed offset = 0 := by
  have h := (Random.randReal_range seed (offset : Int) .bits32 10).2.2
  rw [expected_eq_if, if_neg]
  linarith

theorem expected_of_neg (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat)
    (hp : p < 0) : expected input p seed offset = input offset / (1-p) := by
  have h := (Random.randReal_range seed (offset : Int) .bits32 10).1
  rw [expected_eq_if, if_pos]
  linarith

/-- Exact decoder connection, with no fallback or abstract random callback. -/
theorem canonical_decoder (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat) :
    Random.seededDropoutExpected? input p seed offset =
      some (expected input p seed offset) :=
  Random.seededDropoutExpected_option_eq input p seed offset

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Formula

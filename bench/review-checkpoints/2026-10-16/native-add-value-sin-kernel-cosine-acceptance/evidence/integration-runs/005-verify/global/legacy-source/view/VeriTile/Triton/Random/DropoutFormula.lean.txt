import VeriTile.Triton.Random.Values

/-! Independent input-memory formula using the original runtime seed and offset.
Finiteness is a theorem of the canonical random generator. This pure real formula
does not assert source-program IO, frame, termination, memory safety or FP behavior.
In particular p=1 does not remove the eager FP division from the source program. -/
namespace VeriTile.Triton.Random
noncomputable section

def seededDropoutExpected (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat) : ℝ :=
  if randReal seed (offset : Int) .bits32 10 > p then input offset / (1-p) else 0

/-- Option projection retained as an exact connection to the bit-value decoder. -/
def seededDropoutExpected? (input : Nat → ℝ) (p : ℝ) (seed : Int) (offset : Nat) : Option ℝ :=
  ((rand seed (offset : Int) .bits32 10).toRat?).map fun q =>
    if (q : ℝ) > p then input offset / (1-p) else 0

theorem seededDropoutExpected_option_eq (input : Nat → ℝ) (p : ℝ)
    (seed : Int) (offset : Nat) :
    seededDropoutExpected? input p seed offset = some (seededDropoutExpected input p seed offset) := by
  simp [seededDropoutExpected?, seededDropoutExpected, rand_toRat_randRat, randReal]

theorem seededDropoutExpected_defined (input : Nat → ℝ) (p : ℝ)
    (seed : Int) (offset : Nat) :
    ∃ q : Rat, 0 ≤ q ∧ q < 1 ∧
      seededDropoutExpected? input p seed offset =
        some (if (q : ℝ) > p then input offset / (1-p) else 0) := by
  obtain ⟨q, hq, hzero, _, hone⟩ := canonical_rand_range seed (offset : Int) .bits32 10
  exact ⟨q, hzero, hone, by simp [seededDropoutExpected?, hq]⟩

theorem seededDropoutExpected_total (input : Nat → ℝ) (p : ℝ)
    (seed : Int) (offset : Nat) :
    (seededDropoutExpected? input p seed offset).isSome = Bool.true := by
  rw [seededDropoutExpected_option_eq]
  rfl

theorem seededDropoutExpected_p_one (input : Nat → ℝ) (seed : Int) (offset : Nat) :
    seededDropoutExpected? input 1 seed offset = some 0 := by
  obtain ⟨q, _, hone, heq⟩ := seededDropoutExpected_defined input 1 seed offset
  have hreal : (q : ℝ) < 1 := by exact_mod_cast hone
  simpa [not_lt.mpr (le_of_lt hreal)] using heq

theorem seededDropoutExpected_real_p_one (input : Nat → ℝ) (seed : Int) (offset : Nat) :
    seededDropoutExpected input 1 seed offset = 0 := by
  have h := seededDropoutExpected_p_one input seed offset
  rw [seededDropoutExpected_option_eq] at h
  exact Option.some.inj h

end
end VeriTile.Triton.Random

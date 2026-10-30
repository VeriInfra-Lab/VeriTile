import VeriTile.Triton.Random.Values

/-! The real projection of the existing two-rounding uint32 uniform decoder.
Its finiteness proof selects the actual result; no default value is used. -/
namespace VeriTile.Triton.Random
open RoundBridge Dyadic

def uniformRat (word : Word) : Rat :=
  ((uniform word).toRat?).get (by rw [canonical_uniform_toRat]; rfl)

theorem uniformRat_eq (word : Word) :
    uniformRat word = uniformDyadic (foldSign word) := by
  simp [uniformRat, canonical_uniform_toRat]

theorem uniform_toRat_uniformRat (word : Word) :
    (uniform word).toRat? = some (uniformRat word) := by
  rw [uniformRat_eq]
  exact canonical_uniform_toRat word

noncomputable def uniformReal (word : Word) : ℝ := (uniformRat word : ℝ)

theorem uniformRat_range (word : Word) :
    0 ≤ uniformRat word ∧ uniformRat word ≤ 1-(1:Rat)/2^24 ∧ uniformRat word < 1 := by
  rw [uniformRat_eq]
  exact uniformDyadic_range _ (foldSign_lt _)

end VeriTile.Triton.Random

import VeriTile.Triton.Random.Proofs.CanonicalUniformBridge
import Mathlib.Data.Real.Basic

/-! Total rational and real projections of the canonical random bit value.
The rational projection uses Option.get with a proved finiteness witness;
there is no fallback value or replacement random implementation.
This module does not import Core.Ast or a backend Algebra. -/
namespace VeriTile.Triton.Random
open RoundBridge Dyadic

theorem canonical_rand_toRat (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    (rand seed offset width nRounds).toRat? =
      some (uniformDyadic (foldSign (randint seed offset width nRounds))) :=
  canonical_uniform_toRat _

theorem canonical_rand_range (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    ∃ q : Rat, (rand seed offset width nRounds).toRat? = some q ∧ 0 ≤ q ∧
      q ≤ 1-(1:Rat)/2^24 ∧ q < 1 :=
  canonical_uniform_range _

/-- The actual canonical software result, projected using its finite-value proof. -/
def randRat (seed offset : Int) (width : OffsetWidth := .bits32) (nRounds : Nat := 10) : Rat :=
  ((rand seed offset width nRounds).toRat?).get (by rw [canonical_rand_toRat]; rfl)

theorem randRat_eq (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    randRat seed offset width nRounds =
      uniformDyadic (foldSign (randint seed offset width nRounds)) := by
  simp [randRat, canonical_rand_toRat]

theorem rand_toRat_randRat (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    (rand seed offset width nRounds).toRat? = some (randRat seed offset width nRounds) := by
  rw [randRat_eq]
  exact canonical_rand_toRat seed offset width nRounds

theorem randRat_range (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    0 ≤ randRat seed offset width nRounds ∧
    randRat seed offset width nRounds ≤ 1-(1:Rat)/2^24 ∧
    randRat seed offset width nRounds < 1 := by
  rw [randRat_eq]
  exact uniformDyadic_range _ (foldSign_lt _)

noncomputable def randReal (seed offset : Int) (width : OffsetWidth := .bits32)
    (nRounds : Nat := 10) : ℝ := (randRat seed offset width nRounds : ℝ)

theorem randReal_range (seed offset : Int) (width : OffsetWidth) (nRounds : Nat) :
    0 ≤ randReal seed offset width nRounds ∧
    randReal seed offset width nRounds ≤ 1-(1:ℝ)/2^24 ∧
    randReal seed offset width nRounds < 1 := by
  obtain ⟨hzero, hmax, hone⟩ := randRat_range seed offset width nRounds
  have hmaxReal : (randRat seed offset width nRounds : ℝ) ≤
      ((1-(1:Rat)/2^24 : Rat) : ℝ) := Rat.cast_le.mpr hmax
  push_cast at hmaxReal
  unfold randReal
  exact ⟨by exact_mod_cast hzero, hmaxReal, by exact_mod_cast hone⟩

end VeriTile.Triton.Random

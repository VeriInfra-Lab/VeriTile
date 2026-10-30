import VeriTile.Triton.Random.Proofs.RoundDyadic

namespace VeriTile.Triton.Random

theorem foldSign_lt (x : Word) : foldSign x < 2^31 := by
  have hx := x.isLt
  simp only [foldSign]
  split <;> omega

end VeriTile.Triton.Random

namespace VeriTile.Triton.Random.RoundBridge
open Dyadic

set_option maxRecDepth 3000
set_option maxHeartbeats 2000000

theorem mul_of_positive_decodes (x y : FP.Value .fp32) (a b : Rat)
    (hx : x.decode = .finite false a) (hy : y.decode = .finite false b) :
    FP.mul FP.Config.ieeeValue x y = FP.round .fp32 FP.Config.ieeeValue (a*b) := by
  unfold FP.mul FP.prepare
  simp only [FP.Config.ieeeValue, Bool.false_and, Bool.false_eq_true, ↓reduceIte]
  rw [hx,hy]
  rfl

theorem scale_decode : scale.decode = .finite false ((16777215:Rat)/2^55) := by
  decide +kernel

/-- Both actual FP.round/FP.mul stages are retained. The dyadic expression is
proved equal to their exact decoded result for every 32-bit input word. -/
theorem canonical_uniform_decode (word : Word) :
    (uniform word).decode = .finite false (uniformDyadic (foldSign word)) := by
  let n := foldSign word
  have hn : n < 2^31 := foldSign_lt word
  have hfirst : (FP.round .fp32 FP.Config.ieeeValue (Rat.ofInt (n:Int))).decode =
      .finite false (round24 n:Rat) := by
    simpa only [pow_zero, div_one] using round_dyadic_decode n 0 (by omega) (by decide)
  unfold uniform
  rw [mul_of_positive_decodes _ _ _ _ hfirst scale_decode]
  have harg : (round24 n:Rat)*((16777215:Rat)/2^55) =
      (productNumerator n:Rat)/2^55 := by
    unfold productNumerator
    push_cast
    norm_num
    ring
  rw [harg]
  have hnum : productNumerator n < 2^55 :=
    lt_of_le_of_lt (productNumerator_bound n hn) (by norm_num)
  exact round_dyadic_decode (productNumerator n) 55 hnum (by decide)

theorem canonical_uniform_toRat (word : Word) :
    (uniform word).toRat? = some (uniformDyadic (foldSign word)) := by
  unfold FP.Value.toRat?
  rw [canonical_uniform_decode]
  rfl

/-- Finiteness and range are consequences of the canonical implementation,
with no raw-bound, nonzero-random or substituted-oracle premise. -/
theorem canonical_uniform_range (word : Word) :
    ∃ q : Rat, (uniform word).toRat? = some q ∧ 0 ≤ q ∧
      q ≤ 1-(1:Rat)/2^24 ∧ q < 1 := by
  refine ⟨uniformDyadic (foldSign word), canonical_uniform_toRat word, ?_⟩
  exact uniformDyadic_range _ (foldSign_lt word)

end VeriTile.Triton.Random.RoundBridge

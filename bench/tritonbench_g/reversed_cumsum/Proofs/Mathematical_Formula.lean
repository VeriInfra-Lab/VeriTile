import bench.tritonbench_g.reversed_cumsum.Proofs.Mathematical_RealMath

/-! Precisely when the unspecified tail can be removed from the suffix.
These independent mathematical facts do not alter the source load. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.Formula
open VeriTile.Triton RealMath
open scoped BigOperators
noncomputable section

def paddedInput (input padding : Nat → ℝ) (T : Nat) (k : Nat) : ℝ :=
  if k < T then input k else padding k

def tail (padding : Nat → ℝ) (T padded : Nat) : ℝ :=
  ∑ k ∈ (Finset.range padded).filter (fun k => T ≤ k), padding k

theorem padded_suffix (input padding : Nat → ℝ) (T padded t : Nat)
    (hT : T ≤ padded) (ht : t ≤ T) :
    suffixValue (paddedInput input padding T) padded t =
      suffixValue input T t + tail padding T padded :=
  padding_contribution input padding T padded t hT ht

/-- Cancellation of the total tail, rather than a hidden per-lane default,
is exactly the mathematical condition for the ordinary suffix formula. -/
theorem ordinary_suffix_iff (input padding : Nat → ℝ) (T padded t : Nat)
    (hT : T ≤ padded) (ht : t ≤ T) :
    suffixValue (paddedInput input padding T) padded t = suffixValue input T t ↔
      tail padding T padded = 0 := by
  rw [padded_suffix input padding T padded t hT ht]
  constructor <;> intro h <;> linarith

theorem zero_padding (input padding : Nat → ℝ) (T padded t : Nat)
    (hT : T ≤ padded) (ht : t ≤ T)
    (hz : ∀ k, T ≤ k → k < padded → padding k = 0) :
    suffixValue (paddedInput input padding T) padded t = suffixValue input T t := by
  apply (ordinary_suffix_iff input padding T padded t hT ht).mpr
  apply Finset.sum_eq_zero
  intro k hk
  obtain ⟨hk, hT⟩ := Finset.mem_filter.mp hk
  exact hz k hT (Finset.mem_range.mp hk)

theorem exact_length (input padding : Nat → ℝ) (T t : Nat) (ht : t ≤ T) :
    suffixValue (paddedInput input padding T) T t = suffixValue input T t := by
  apply zero_padding input padding T T t le_rfl ht
  intro k hk hk'
  omega

/-- One nonzero time-tail lane contaminates every preceding valid suffix. -/
theorem five_rows_nonzero_tail (t : Nat) (ht : t ≤ 5) :
    suffixValue (paddedInput (fun _ => 2) (fun _ => 7) 5) 6 t =
      suffixValue (fun _ => 2) 5 t + 7 := by
  rw [padded_suffix _ _ 5 6 t (by decide) ht]
  have htail : (Finset.range 6).filter (fun k => 5 ≤ k) = {5} := by decide
  simp [tail, htail]

theorem first_and_last_values :
    suffixValue (paddedInput (fun _ => 2) (fun _ => 7) 5) 6 0 = 17 ∧
    suffixValue (paddedInput (fun _ => 2) (fun _ => 7) 5) 6 4 = 9 := by
  constructor
  · rw [five_rows_nonzero_tail 0 (by decide)]
    norm_num [suffixValue, initialSum, Finset.sum_range_succ]
  · rw [five_rows_nonzero_tail 4 (by decide)]
    norm_num [suffixValue, initialSum, Finset.sum_range_succ]

#axiomsClean ordinary_suffix_iff
#axiomsClean zero_padding
#axiomsClean exact_length
#axiomsClean five_rows_nonzero_tail
#axiomsClean first_and_last_values
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.Formula

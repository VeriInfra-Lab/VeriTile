import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent real formula for the complete reverse scan. The mathematical
upper bound is the padded length: every unchecked tail value contributes to
the carry before any output is written. No padding convention is imposed. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

def initialSum (f : Nat → ℝ) (n : Nat) : ℝ := ∑ k ∈ Finset.range n, f k

def suffix (f : Nat → ℝ) (n t : Nat) : ℝ := initialSum f n - initialSum f t

def carry (f : Nat → ℝ) (BT chunks c : Nat) : ℝ := suffix f (chunks * BT) (c * BT)

def padded (input padding : Nat → ℝ) (T k : Nat) : ℝ :=
  if k < T then input k else padding k

def tail (padding : Nat → ℝ) (T n : Nat) : ℝ :=
  ∑ k ∈ (Finset.range n).filter (fun k => T ≤ k), padding k

theorem initialSum_add (f : Nat → ℝ) (n m : Nat) :
    initialSum f (n + m) = initialSum f n + initialSum (fun k => f (n + k)) m := by
  exact Finset.sum_range_add f n m

theorem initialSum_succ (f : Nat → ℝ) (n : Nat) :
    initialSum f (n + 1) = initialSum f n + f n := by
  exact Finset.sum_range_succ f n

theorem suffix_filter (f : Nat → ℝ) (n t : Nat) (ht : t ≤ n) :
    suffix f n t = ∑ k ∈ (Finset.range n).filter (fun k => t ≤ k), f k := by
  have hsmall : (Finset.range n).filter (fun k => ¬t ≤ k) = Finset.range t := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_range]
    omega
  have hsplit := Finset.sum_filter_add_sum_filter_not (Finset.range n) (fun k => t ≤ k) f
  rw [hsmall] at hsplit
  unfold suffix initialSum
  linarith

theorem carry_end (f : Nat → ℝ) (BT chunks : Nat) : carry f BT chunks chunks = 0 := by
  simp [carry, suffix]

theorem carry_predecessor (f : Nat → ℝ) (BT chunks c : Nat) :
    carry f BT chunks (c + 1) + ∑ k : Fin BT, f (c * BT + k.val) = carry f BT chunks c := by
  rw [Fin.sum_univ_eq_sum_range (fun k => f (c * BT + k)) BT]
  unfold carry suffix
  rw [Nat.add_mul, Nat.one_mul, initialSum_add]
  unfold initialSum
  ring

theorem prefix_finite (g : Nat → ℝ) (BT : Nat) (i : Fin BT) :
    (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), g k.val) =
      initialSum g (i.val + 1) := by
  apply Finset.sum_bij (fun k _ => k.val)
  · intro k hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    exact Finset.mem_range.mpr (by omega)
  · intro a _ b _ he
    exact Fin.ext he
  · intro k hk
    have hk' := Finset.mem_range.mp hk
    exact ⟨⟨k, by omega⟩, by simp only [Finset.mem_filter, Finset.mem_univ, true_and]; omega, rfl⟩
  · intro k hk
    rfl

/-- The scalar source subtracts a forward inclusive prefix and adds the carry
after incorporating the current chunk. This produces the padded suffix. -/
theorem scalar_from_chunk (f : Nat → ℝ) (BT chunks c : Nat) (i : Fin BT) :
    f (c * BT + i.val) -
        (∑ k ∈ Finset.univ.filter (fun k : Fin BT => k.val ≤ i.val), f (c * BT + k.val)) +
        carry f BT chunks c = suffix f (chunks * BT) (c * BT + i.val) := by
  rw [prefix_finite (fun k => f (c * BT + k)) BT i, initialSum_succ]
  unfold carry suffix
  rw [initialSum_add]
  ring

/-- Every valid output includes exactly the same unspecified tail sum. -/
theorem padding_contribution (input padding : Nat → ℝ) (T n t : Nat)
    (hT : T ≤ n) (ht : t ≤ T) :
    suffix (padded input padding T) n t = suffix input T t + tail padding T n := by
  have hprefix (m : Nat) (hm : m ≤ T) :
      initialSum (padded input padding T) m = initialSum input m := by
    apply Finset.sum_congr rfl
    intro k hk
    simp only [padded, if_pos (lt_of_lt_of_le (Finset.mem_range.mp hk) hm)]
  have htail : suffix (padded input padding T) n T = tail padding T n := by
    rw [suffix_filter _ _ _ hT]
    apply Finset.sum_congr rfl
    intro k hk
    simp [padded, Nat.not_lt.mpr (Finset.mem_filter.mp hk).2]
  rw [← htail]
  unfold suffix
  rw [hprefix T le_rfl, hprefix t ht]
  ring

/-- Zero total tail contribution is the exact real condition for the usual
suffix specification at any valid index. Individual tail lanes need not vanish. -/
theorem ordinary_suffix_iff (input padding : Nat → ℝ) (T n t : Nat)
    (hT : T ≤ n) (ht : t ≤ T) :
    suffix (padded input padding T) n t = suffix input T t ↔ tail padding T n = 0 := by
  rw [padding_contribution input padding T n t hT ht]
  constructor <;> intro h <;> linarith

theorem tail_zero_of_padding_zero (padding : Nat → ℝ) (T n : Nat)
    (hp : ∀ k, T ≤ k → k < n → padding k = 0) : tail padding T n = 0 := by
  apply Finset.sum_eq_zero
  intro k hk
  obtain ⟨hn, hT⟩ := Finset.mem_filter.mp hk
  exact hp k hT (Finset.mem_range.mp hn)

theorem tail_zero_of_aligned (padding : Nat → ℝ) (T : Nat) : tail padding T T = 0 := by
  apply Finset.sum_eq_zero
  intro k hk
  obtain ⟨hk, hT⟩ := Finset.mem_filter.mp hk
  have := Finset.mem_range.mp hk
  omega

/-- A concrete mathematical obstruction, without claiming zero padding for
the original boundary load: one valid zero followed by an unspecified one. -/
theorem unspecified_tail_changes_output :
    suffix (padded (fun _ => 0) (fun _ => 1) 1) 2 0 = 1 ∧
      suffix (fun _ => 0) 1 0 = 0 := by
  norm_num [suffix, initialSum, padded, Finset.sum_range_succ]

#axiomsClean suffix_filter
#axiomsClean carry_predecessor
#axiomsClean scalar_from_chunk
#axiomsClean padding_contribution
#axiomsClean ordinary_suffix_iff
#axiomsClean tail_zero_of_padding_zero
#axiomsClean tail_zero_of_aligned
#axiomsClean unspecified_tail_changes_output
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.Formula

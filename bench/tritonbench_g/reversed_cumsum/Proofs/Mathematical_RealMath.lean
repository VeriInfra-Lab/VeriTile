import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Independent finite suffix sums. The upper limit is the padded length,
so an unspecified time tail remains part of the mathematical result. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath
open VeriTile.Triton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000

def initialSum (f : Nat → ℝ) (n : Nat) : ℝ := ∑ k ∈ Finset.range n, f k
def suffixValue (f : Nat → ℝ) (n t : Nat) : ℝ := initialSum f n - initialSum f t
def carry (f : Nat → ℝ) (BT chunks c : Nat) : ℝ := suffixValue f (chunks * BT) (c * BT)

theorem initialSum_add (f : Nat → ℝ) (n m : Nat) :
    initialSum f (n + m) = initialSum f n + initialSum (fun k => f (n + k)) m := by
  exact Finset.sum_range_add f n m

theorem suffix_filter (f : Nat → ℝ) (n t : Nat) (ht : t ≤ n) :
    suffixValue f n t = ∑ k ∈ (Finset.range n).filter (fun k => t ≤ k), f k := by
  have hsmall : (Finset.range n).filter (fun k => ¬t ≤ k) = Finset.range t := by
    ext k
    simp only [Finset.mem_filter, Finset.mem_range]
    omega
  have hsplit := Finset.sum_filter_add_sum_filter_not (Finset.range n) (fun k => t ≤ k) f
  rw [hsmall] at hsplit
  unfold suffixValue initialSum
  linarith

theorem carry_end (f : Nat → ℝ) (BT chunks : Nat) : carry f BT chunks chunks = 0 := by
  simp [carry, suffixValue]

theorem carry_predecessor (f : Nat → ℝ) (BT chunks c : Nat) :
    carry f BT chunks (c + 1) + ∑ k : Fin BT, f (c * BT + k.val) = carry f BT chunks c := by
  rw [Fin.sum_univ_eq_sum_range (fun k => f (c * BT + k)) BT]
  unfold carry suffixValue
  rw [Nat.add_mul, Nat.one_mul, initialSum_add]
  unfold initialSum
  ring

theorem local_suffix (f : Nat → ℝ) (BT c : Nat) (i : Fin BT) :
    (∑ k ∈ Finset.univ.filter (fun k : Fin BT => i.val ≤ k.val), f (c * BT + k.val)) =
      initialSum (fun k => f (c * BT + k)) BT - initialSum (fun k => f (c * BT + k)) i.val := by
  rw [← suffixValue, suffix_filter _ _ _ (Nat.le_of_lt i.isLt)]
  apply Finset.sum_bij (fun k _ => k.val)
  · intro k hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk
    exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr k.isLt, hk⟩
  · intro a _ b _ he
    exact Fin.ext he
  · intro k hk
    obtain ⟨hk, hi⟩ := Finset.mem_filter.mp hk
    refine ⟨⟨k, Finset.mem_range.mp hk⟩, ?_, rfl⟩
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, hi⟩
  · intro k hk
    rfl

theorem suffix_from_chunk (f : Nat → ℝ) (BT chunks c : Nat) (i : Fin BT) :
    carry f BT chunks (c + 1) +
      (∑ k ∈ Finset.univ.filter (fun k : Fin BT => i.val ≤ k.val), f (c * BT + k.val)) =
        suffixValue f (chunks * BT) (c * BT + i.val) := by
  rw [local_suffix]
  unfold carry suffixValue
  rw [Nat.add_mul, Nat.one_mul, initialSum_add, initialSum_add]
  ring

/-- Separate the real input suffix from the explicit padding contribution. -/
theorem padding_contribution (input padding : Nat → ℝ) (T n t : Nat)
    (hT : T ≤ n) (ht : t ≤ T) :
    suffixValue (fun k => if k < T then input k else padding k) n t =
      suffixValue input T t +
        ∑ k ∈ (Finset.range n).filter (fun k => T ≤ k), padding k := by
  let f := fun k => if k < T then input k else padding k
  have hprefix (m : Nat) (hm : m ≤ T) : initialSum f m = initialSum input m := by
    apply Finset.sum_congr rfl
    intro k hk
    simp only [f, if_pos (lt_of_lt_of_le (Finset.mem_range.mp hk) hm)]
  have htail : suffixValue f n T = ∑ k ∈ (Finset.range n).filter (fun k => T ≤ k), padding k := by
    rw [suffix_filter _ _ _ hT]
    apply Finset.sum_congr rfl
    intro k hk
    simp [f, Nat.not_lt.mpr (Finset.mem_filter.mp hk).2]
  change suffixValue f n t = _
  rw [← htail]
  unfold suffixValue
  rw [hprefix T le_rfl, hprefix t ht]
  ring

def upperTriTile (BT : Nat) : Tile .real [BT,BT] :=
  Tile.mat fun i j => some (if i.val ≤ j.val then (1 : ℝ) else 0)

theorem upper_triangular_dot (BT BS : Nat) (f : Fin BT → Fin BS → ℝ)
    (i : Fin BT) (j : Fin BS) :
    (Tile.dot [] (upperTriTile BT) (Tile.mat fun k e => some (f k e))).data (i,j,PUnit.unit) =
      some (∑ k ∈ Finset.univ.filter (fun k : Fin BT => i.val ≤ k.val), f k j) := by
  rw [Tile.dot_nil_data]
  simp only [upperTriTile, Tile.mat, Option.map₂_some_some]
  have hterm (k : Fin BT) :
      ((if i.val ≤ k.val then (1 : ℝ) else 0) * f k j) =
        if i.val ≤ k.val then f k j else 0 := by split <;> simp
  simp only [hterm]
  change (∑ k : Fin BT, (((if i.val ≤ k.val then f k j else 0) : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum, Finset.sum_filter]
  rfl

#axiomsClean suffix_filter
#axiomsClean carry_predecessor
#axiomsClean suffix_from_chunk
#axiomsClean padding_contribution
#axiomsClean upper_triangular_dot
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealMath

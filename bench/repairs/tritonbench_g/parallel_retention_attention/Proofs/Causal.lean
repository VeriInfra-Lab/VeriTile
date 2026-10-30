import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.GraphReal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

theorem causal_split (g : ℝ) (x : Nat → ℝ) (base L r : Nat) (hr : r<L) :
    (∑ n∈Finset.range base,g^(base+r-n)*x n)+
      (∑ n∈Finset.range L,if n≤r then g^(r-n)*x (base+n) else 0)=
      ∑ n∈Finset.range (base+r+1),g^(base+r-n)*x n := by
  have diagonal : (∑ n∈Finset.range L,if n≤r then g^(r-n)*x (base+n) else 0)=
      ∑ n∈Finset.range (r+1),g^(r-n)*x (base+n) := by
    have eq : (fun n => if n≤r then g^(r-n)*x (base+n) else 0)=
        (fun n => if n<r+1 then g^(r-n)*x (base+n) else 0) := by
      funext n
      simp only [Nat.lt_succ_iff]
    rw [eq]
    exact sum_padded (r+1) L (by omega) _
  rw [diagonal,show base+r+1=base+(r+1) by omega]
  conv_rhs => rw [Finset.sum_range_add]
  congr 1
  apply Finset.sum_congr rfl
  intro n hn
  rw [Nat.add_sub_add_left]

theorem forward_causal (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r v : Nat) (hr : r<128) :
    forwardLane realOps p s x tile r v=
      ∑ n∈Finset.range (tile.chunk*128+r+1),gamma (tile.bh%s.H)^(tile.chunk*128+r-n)*forwardProduct s x tile r v n := by
  rw [forward_phase_sums]
  exact causal_split _ _ _ _ _ hr

/-- Extending a causal prefix to the logical sequence size introduces only
masked zero terms, not assumptions about the input values. -/
theorem causal_to_logical (g : ℝ) (x : Nat → ℝ) (T t : Nat) (ht : t<T) :
    (∑ n∈Finset.range (t+1),g^(t-n)*x n)=
      ∑ n∈Finset.range T,if n≤t then g^(t-n)*x n else 0 := by
  symm
  have eq : (fun n => if n≤t then g^(t-n)*x n else 0)=
      (fun n => if n<t+1 then g^(t-n)*x n else 0) := by
    funext n
    simp only [Nat.lt_succ_iff]
  rw [eq]
  exact sum_padded (t+1) T (by omega) _

#axiomsClean causal_split
#axiomsClean forward_causal
#axiomsClean causal_to_logical
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention

import bench.tritonbench_g.context_attn_nopad.Proofs.HostBounds

namespace VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host
open VeriTile.Triton Formula Program
noncomputable section
set_option maxHeartbeats 800000

private theorem sum_prefix (f : Nat → ℝ) (n m : Nat) (hm : n ≤ m)
    (hz : ∀ j, n ≤ j → j < m → f j = 0) :
    (∑ j : Fin m, f j.val) = ∑ j ∈ Finset.range n, f j := by
  rw [Fin.sum_univ_eq_sum_range]
  symm
  apply Finset.sum_subset (Finset.range_mono hm)
  intro j hj hn
  exact hz j (by simpa using hn) (by simpa using hj)

variable {c : Call} {r : Regions} {A : FlatAlloc} {s : BlockState}

theorem count_active (hm : Memory c r A s) (idx : GridIndex c.grid) (i : RealIO.Lane c.layout)
    (hi : RealIO.active c.layout (c.launch idx) (inputs s r) i) :
    count c.layout (c.launch idx) (inputs s r) = (c.launch idx).block + 1 := by
  have ht := (active_iff hm idx i).mp hi
  have hs : ((c.launch idx).block * c.layout.BM : Int) <
      Formula.sequence c.layout (c.launch idx) (inputs s r) := by
    change ((c.launch idx).block * 128 : Int) < (inputs s r).sequence _
    rw [hm.length _ (launch_batch_lt c idx)]
    dsimp [queryRow, Call.layout] at ht
    omega
  rw [Formula.count, if_pos hs]
  simp [Call.layout]

theorem score_logical (hm : Memory c r A s) (idx : GridIndex c.grid)
    (i : RealIO.Lane c.layout)
    (hi : RealIO.active c.layout (c.launch idx) (inputs s r) i)
    (j : Nat) (hj : j ≤ queryRow c.layout (c.launch idx) i.1.val) :
    Formula.score c.layout (c.launch idx) (inputs s r) c.scale i.1.val j =
      c.score (inputs s r) (c.launch idx).batch (c.launch idx).head
        (queryRow c.layout (c.launch idx) i.1.val) j := by
  have ht := (active_iff hm idx i).mp hi
  have hk : (j : Int) < Formula.sequence c.layout (c.launch idx) (inputs s r) := by
    rw [Formula.sequence, hm.length _ (launch_batch_lt c idx)]
    omega
  unfold RealIO.active at hi
  simp only [Formula.score, Formula.query, Formula.key, if_pos hi, if_pos hk,
    hm.start _ (launch_batch_lt c idx)]
  simp only [Call.score, Call.layout, Tensor3.offset, ← Int.natCast_add, ← Int.natCast_mul, Int.toNat_natCast]
  rfl

theorem value_logical (hm : Memory c r A s) (idx : GridIndex c.grid)
    (i : RealIO.Lane c.layout)
    (hi : RealIO.active c.layout (c.launch idx) (inputs s r) i)
    (j : Nat) (hj : j ≤ queryRow c.layout (c.launch idx) i.1.val) :
    Formula.value c.layout (c.launch idx) (inputs s r) j i.2.1.val =
      (inputs s r).v (c.v.offset (c.start (c.launch idx).batch+j) (c.launch idx).head i.2.1.val) := by
  have ht := (active_iff hm idx i).mp hi
  have hk : (j : Int) < Formula.sequence c.layout (c.launch idx) (inputs s r) := by
    rw [Formula.sequence, hm.length _ (launch_batch_lt c idx)]
    omega
  rw [Formula.value, if_pos hk]
  simp only [hm.start _ (launch_batch_lt c idx), Call.layout,
    Tensor3.offset, ← Int.natCast_add, ← Int.natCast_mul, Int.toNat_natCast]

/-- All visited lanes after the logical query have exactly zero weight because
the original sentinel is negative infinity. No finite-sentinel approximation is
used here. Every key at or before the query is inside the loaded sequence. -/
theorem closedExpected_logical (hm : Memory c r A s) (idx : GridIndex c.grid)
    (i : RealIO.Lane c.layout)
    (hi : RealIO.active c.layout (c.launch idx) (inputs s r) i) :
    RealIO.closedExpected c.layout c.scale (c.launch idx) (inputs s r) i =
      c.expected (inputs s r) (c.launch idx).batch (c.launch idx).head
        (queryRow c.layout (c.launch idx) i.1.val) i.2.1.val := by
  have hc := count_active hm idx i hi
  have hiB : i.1.val < 128 := i.1.isLt
  have hprefix : queryRow c.layout (c.launch idx) i.1.val + 1 ≤
      count c.layout (c.launch idx) (inputs s r) * c.layout.BN := by
    rw [hc]
    dsimp [queryRow, Call.layout]
    omega
  have hwzero : ∀ j, queryRow c.layout (c.launch idx) i.1.val + 1 ≤ j →
      Formula.weight c.layout (c.launch idx) (inputs s r) c.scale i.1.val j = 0 := by
    intro j hj
    exact if_neg (by omega)
  unfold RealIO.closedExpected Call.expected
  rw [sum_prefix (fun j => Formula.weight c.layout (c.launch idx) (inputs s r) c.scale i.1.val j *
      Formula.value c.layout (c.launch idx) (inputs s r) j i.2.1.val) _ _ hprefix
      (by intro j hj _; simp only [hwzero j hj, zero_mul]),
    sum_prefix (fun j => Formula.weight c.layout (c.launch idx) (inputs s r) c.scale i.1.val j) _ _ hprefix
      (by intro j hj _; exact hwzero j hj)]
  congr 1
  · apply Finset.sum_congr rfl
    intro j hj
    have hj' : j ≤ queryRow c.layout (c.launch idx) i.1.val := by
      have := Finset.mem_range.mp hj
      omega
    rw [Formula.weight, if_pos hj', score_logical hm idx i hi j hj', value_logical hm idx i hi j hj']
  · apply Finset.sum_congr rfl
    intro j hj
    have hj' : j ≤ queryRow c.layout (c.launch idx) i.1.val := by
      have := Finset.mem_range.mp hj
      omega
    rw [Formula.weight, if_pos hj', score_logical hm idx i hi j hj']

end
end VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host

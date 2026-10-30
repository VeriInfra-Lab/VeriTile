import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeIndex

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole
open VeriTile.Triton Math Program Initialization
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1500000

/-- Logical tensor coordinates, independent of tile dimensions and launch IDs.
The output includes chunk padding. Prepared padding is zero. -/
def logicalPrepared (c : Host) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k t : Nat) : ℝ :=
  if h < c.heads ∧ t < c.chunk ∧ k*c.chunk+t < c.sequence then
    let x := a.dt (b*c.dtBatch+(k*c.chunk+t)*c.dtTime+h*c.dtHead) +
      if hasBias then a.bias (h*c.biasHead) else 0
    let y := if useSoftplus then (if x ≤ 20 then Real.log (1+Real.exp x) else x) else x
    upper.cap (max y lower)
  else 0

/-- Inclusive mathematical prefix. The sum has j+1 terms; it does not refer
to BLOCK_SIZE_CHUNK, scan internals, or the candidate source. -/
def logicalCumulative (c : Host) (a : Inputs) (hasBias useSoftplus : Bool)
    (lower : ℝ) (upper : UpperBound) (b h k j : Nat) : ℝ :=
  ∑ t ∈ Finset.range (j+1), logicalPrepared c a hasBias useSoftplus lower upper b h k t *
    a.rate (h*c.rateHead)

theorem prepared_logical (c : Host) (l : Launch) (hl : c.valid l) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound) (i j : Nat) :
    prepared c.layout l a hasBias useSoftplus lower upper i j =
      logicalPrepared c a hasBias useSoftplus lower upper l.batch (head c.layout l i) l.chunk j := by
  have hs := chunk_start_lt c l hl
  have hm : j < min c.chunk (c.sequence-l.chunk*c.chunk) ↔
      j < c.chunk ∧ l.chunk*c.chunk+j < c.sequence := by omega
  have ha : dtAddress c.layout l i j =
      l.batch*c.dtBatch+(l.chunk*c.chunk+j)*c.dtTime+(head c.layout l i)*c.dtHead := by
    simp [dtAddress,Host.layout]
    ring
  have hm' : activeInput c.layout l i j ↔ head c.layout l i < c.heads ∧
      j < c.chunk ∧ l.chunk*c.chunk+j < c.sequence := by
    change (head c.layout l i < c.heads ∧ j < min c.chunk (c.sequence-l.chunk*c.chunk)) ↔ _
    rw [hm]
  simp only [prepared,logicalPrepared,hm',ha,softplus]
  rfl

theorem prefix_range (n j : Nat) (hj : j < n) (f : Nat → ℝ) :
    (∑ t ∈ Finset.range n, if t ≤ j then f t else 0) = ∑ t ∈ Finset.range (j+1), f t := by
  have sub : Finset.range (j+1) ⊆ Finset.range n := Finset.range_mono (by omega)
  calc
    _ = ∑ t ∈ Finset.range (j+1), if t ≤ j then f t else 0 := by
      symm
      apply Finset.sum_subset sub
      intro t _ hn
      have ht : ¬ t ≤ j := by simpa using hn
      simp [ht]
    _ = _ := by
      apply Finset.sum_congr rfl
      intro t ht
      have ht' : t ≤ j := by simpa using ht
      simp [ht']

theorem cumulative_logical (c : Host) (l : Launch) (hl : c.valid l) (a : Inputs)
    (hasBias useSoftplus : Bool) (lower : ℝ) (upper : UpperBound)
    (i : Fin c.layout.BH) (j : Fin c.layout.BC)
    (hi : head c.layout l i.val < c.heads) :
    cumulative c.layout l a hasBias useSoftplus lower upper i j =
      logicalCumulative c a hasBias useSoftplus lower upper
        l.batch (head c.layout l i.val) l.chunk j.val := by
  have hr : rate c.layout l a i.val = a.rate (head c.layout l i.val*c.rateHead) := by
    exact if_pos hi
  unfold cumulative logicalCumulative
  simp only [prepared_logical c l hl,hr]
  let f : Nat → ℝ := fun t => logicalPrepared c a hasBias useSoftplus lower upper
    l.batch (head c.layout l i.val) l.chunk t * a.rate (head c.layout l i.val*c.rateHead)
  exact (Fin.sum_univ_eq_sum_range (fun t => if t ≤ j.val then f t else 0) c.layout.BC).trans
    (prefix_range c.layout.BC j.val j.isLt f)

end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.Whole

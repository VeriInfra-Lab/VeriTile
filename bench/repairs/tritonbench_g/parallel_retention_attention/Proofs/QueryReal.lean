import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Causal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

def queryProduct (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e n : Nat) : ℝ :=
  (∑ v∈Finset.range s.BV,
      load realOps s.T s.DV x.g tile.bh (tile.chunk*64+r) (tile.value*s.BV+v)*
      load realOps s.T s.DV x.v tile.bh n (tile.value*s.BV+v))*
    load realOps s.T s.DK x.k tile.bh n (tile.key*s.BK+e)

theorem query_previous_step (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e j : Nat) (carry : ℝ) :
    queryStep realOps p s x tile r e false j carry=
      carry*gamma (tile.bh%s.H)^32+
      ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(32-u)*queryProduct s x tile r e (j*32+u) := by
  unfold queryStep
  simp only [Bool.false_eq_true,if_false,zero_add]
  simp_rw [dot_real,decayNat_real,blockDecay_real]
  simp only [real_zero,real_mul,real_cast,real_scale,zero_add]
  congr 1
  apply Finset.sum_congr rfl
  intro u hu
  unfold queryProduct
  ring

theorem query_diagonal_step (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e j : Nat) (carry : ℝ) :
    queryStep realOps p s x tile r e true j carry=
      carry+(Real.sqrt (s.DK:ℝ))⁻¹*(∑ u∈Finset.range 32,
        if j*32+u≤r then gamma (tile.bh%s.H)^(r-(j*32+u))*queryProduct s x tile r e (tile.chunk*64+j*32+u) else 0) := by
  unfold queryStep
  simp only [if_true]
  simp_rw [dot_real,diagonalDecay_real]
  simp only [real_zero,real_mul,real_cast,real_scale,zero_add]
  rw [Finset.mul_sum]
  congr 1
  apply Finset.sum_congr rfl
  intro u hu
  split_ifs with hc
  · unfold queryProduct
    ring
  · ring

theorem query_phase_sums (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e : Nat) :
    queryLane realOps p s x tile r e=(Real.sqrt (s.DK:ℝ))⁻¹*
      ((∑ n∈Finset.range (tile.chunk*64),gamma (tile.bh%s.H)^(tile.chunk*64+r-n)*queryProduct s x tile r e n)+
       ∑ n∈Finset.range 64,if n≤r then gamma (tile.bh%s.H)^(r-n)*queryProduct s x tile r e (tile.chunk*64+n) else 0) := by
  unfold queryLane
  have hp : queryStep realOps p s x tile r e false=
      (fun j carry => carry*gamma (tile.bh%s.H)^32+∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(32-u)*queryProduct s x tile r e (j*32+u)) := by
    funext j carry
    exact query_previous_step p s x tile r e j carry
  have hd : queryStep realOps p s x tile r e true=
      (fun j carry => carry+(Real.sqrt (s.DK:ℝ))⁻¹*∑ u∈Finset.range 32,if j*32+u≤r then gamma (tile.bh%s.H)^(r-(j*32+u))*queryProduct s x tile r e (tile.chunk*64+(j*32+u)) else 0) := by
    funext j carry
    simpa [Nat.add_assoc] using query_diagonal_step p s x tile r e j carry
  rw [hp,hd]
  simp only [real_zero,real_mul,real_cast,real_scale]
  rw [foldN_left,decayNat_real,foldN_sum,← Finset.mul_sum,← mul_assoc,
    previous_blocks _ _ _ _ _ (by norm_num)]
  have he : tile.chunk*2*32=tile.chunk*64 := by omega
  rw [he,sum_tiles 2 32 (fun n => if n≤r then gamma (tile.bh%s.H)^(r-n)*queryProduct s x tile r e (tile.chunk*64+n) else 0)]
  ring

theorem query_causal (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e : Nat) (hr : r<64) :
    queryLane realOps p s x tile r e=(Real.sqrt (s.DK:ℝ))⁻¹*
      ∑ n∈Finset.range (tile.chunk*64+r+1),gamma (tile.bh%s.H)^(tile.chunk*64+r-n)*queryProduct s x tile r e n := by
  rw [query_phase_sums,causal_split _ _ _ _ _ hr]

#axiomsClean query_previous_step
#axiomsClean query_diagonal_step
#axiomsClean query_phase_sums
#axiomsClean query_causal
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention

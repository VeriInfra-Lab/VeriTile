import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Graph
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Real

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

theorem real_zero : realOps.zero=(0:ℝ) := rfl
theorem real_mul (x y : ℝ) : realOps.mul x y=x*y := rfl
theorem real_cast (x : ℝ) : realOps.cast x=x := rfl
theorem real_scale (n : Nat) : realOps.scale n=(Real.sqrt (n:ℝ))⁻¹ := rfl

theorem dot_real (p : Plans) (site : List Nat) (n : Nat) (x y : Nat → ℝ) (carry : ℝ) :
    dot realOps p site n x y carry=carry+∑ i∈Finset.range n,x i*y i :=
  DotPlan.real_sum _ _ (p.complete site n) x y carry

theorem decayInt_nat (h row col : Nat) (hc : col≤row) :
    decayInt realOps (logDecay realOps h) ((row:Int)-(col:Int))=gamma h^(row-col) := by
  have eq : ((row:Int)-(col:Int))=((row-col:Nat):Int) := by omega
  rw [eq]
  have e : decayInt realOps (logDecay realOps h) ((row-col:Nat):Int)=decayNat realOps (logDecay realOps h) (row-col) := by
    simp [decayInt,decayNat,realOps]
  rw [e,decayNat_real]

theorem diagonalDecay_real (h row col : Nat) :
    diagonalDecay realOps (logDecay realOps h) row col=
      if col≤row then gamma h^(row-col) else 0 := by
  unfold diagonalDecay
  split_ifs with hc
  · exact decayInt_nat h row col hc
  · rfl

theorem foldN_left (g : ℝ) (x : Nat → ℝ) (n : Nat) :
    foldN (fun i z => z*g+x i) n 0=leftScan g x n := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,leftScan,ih]

theorem foldN_sum (x : Nat → ℝ) (n : Nat) (carry : ℝ) :
    foldN (fun i z => z+x i) n carry=carry+∑ i∈Finset.range n,x i := by
  induction n with
  | zero => simp [foldN]
  | succ n ih => simp [foldN,ih,Finset.sum_range_succ,add_assoc]

/-- Independent products for a forward shard. Its time variable ranges over
logical positions; load padding remains explicit in both feature factors. -/
def forwardProduct (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v n : Nat) : ℝ :=
  (∑ e∈Finset.range s.BK,
      (load realOps s.T s.DK x.q tile.bh (tile.chunk*128+r) (tile.key*s.BK+e)*(Real.sqrt (s.DK:ℝ))⁻¹)*
      load realOps s.T s.DK x.k tile.bh n (tile.key*s.BK+e))*
    load realOps s.T s.DV x.v tile.bh n (tile.value*s.BV+v)

theorem forward_previous_step (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r v j : Nat) (carry : ℝ) :
    forwardStep realOps p s x tile r v false j carry=
      carry*gamma (tile.bh%s.H)^32+
      ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(32-u)*forwardProduct s x tile r v (j*32+u) := by
  unfold forwardStep
  simp only [Bool.false_eq_true,if_false,zero_add]
  simp_rw [dot_real,decayNat_real,blockDecay_real]
  simp only [real_zero,real_mul,real_cast,real_scale,zero_add]
  congr 1
  apply Finset.sum_congr rfl
  intro u hu
  unfold forwardProduct
  ring

theorem forward_diagonal_step (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r v j : Nat) (carry : ℝ) :
    forwardStep realOps p s x tile r v true j carry=
      carry+∑ u∈Finset.range 32,
        if j*32+u≤r then gamma (tile.bh%s.H)^(r-(j*32+u))*forwardProduct s x tile r v (tile.chunk*128+j*32+u) else 0 := by
  unfold forwardStep
  simp only [if_true]
  simp_rw [dot_real,diagonalDecay_real]
  simp only [real_zero,real_mul,real_cast,real_scale,zero_add]
  congr 1
  apply Finset.sum_congr rfl
  intro u hu
  split_ifs with hc
  · unfold forwardProduct
    ring
  · ring

theorem forward_phase_sums (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r v : Nat) :
    forwardLane realOps p s x tile r v=
      (∑ n∈Finset.range (tile.chunk*128),gamma (tile.bh%s.H)^(tile.chunk*128+r-n)*forwardProduct s x tile r v n)+
      ∑ n∈Finset.range 128,if n≤r then gamma (tile.bh%s.H)^(r-n)*forwardProduct s x tile r v (tile.chunk*128+n) else 0 := by
  unfold forwardLane
  have hp : forwardStep realOps p s x tile r v false=
      (fun j carry => carry*gamma (tile.bh%s.H)^32+∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(32-u)*forwardProduct s x tile r v (j*32+u)) := by
    funext j carry
    exact forward_previous_step p s x tile r v j carry
  have hd : forwardStep realOps p s x tile r v true=
      (fun j carry => carry+∑ u∈Finset.range 32,if j*32+u≤r then gamma (tile.bh%s.H)^(r-(j*32+u))*forwardProduct s x tile r v (tile.chunk*128+(j*32+u)) else 0) := by
    funext j carry
    simpa [Nat.add_assoc] using forward_diagonal_step p s x tile r v j carry
  rw [hp,hd]
  simp only [real_zero,real_mul,real_cast]
  rw [foldN_left,decayNat_real]
  change foldN _ 4 (_*gamma (tile.bh%s.H)^r)=_
  rw [foldN_sum,previous_blocks _ _ _ _ _ (by norm_num)]
  have he : tile.chunk*4*32=tile.chunk*128 := by omega
  rw [he,sum_tiles 4 32 (fun n => if n≤r then gamma (tile.bh%s.H)^(r-n)*forwardProduct s x tile r v (tile.chunk*128+n) else 0)]

#axiomsClean dot_real
#axiomsClean diagonalDecay_real
#axiomsClean forward_previous_step
#axiomsClean forward_diagonal_step
#axiomsClean forward_phase_sums
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention

import bench.tritonbench_g.parallel_retention_attention.Proofs.GraphReal

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

def keyProduct (s : Shape) (x : Inputs ℝ) (tile : Tile) (r e t : Nat) : ℝ :=
  (∑ v∈Finset.range s.BV,
      load realOps s.T s.DV x.v tile.bh (tile.chunk*64+r) (tile.value*s.BV+v)*
      load realOps s.T s.DV x.g tile.bh t (tile.value*s.BV+v))*
    load realOps s.T s.DK x.q tile.bh t (tile.key*s.BK+e)

def valueProduct (s : Shape) (x : Inputs ℝ) (tile : Tile) (r v t : Nat) : ℝ :=
  (∑ e∈Finset.range s.BK,
      load realOps s.T s.DK x.k tile.bh (tile.chunk*64+r) (tile.key*s.BK+e)*
      load realOps s.T s.DK x.q tile.bh t (tile.key*s.BK+e))*
    load realOps s.T s.DV x.g tile.bh t (tile.value*s.BV+v)

theorem keyValue_future_step (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e v j : Nat) (carry : ℝ×ℝ) :
    keyValueFuture realOps p s x tile r e v j carry=
      (carry.1*gamma (tile.bh%s.H)^32+
         ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^u*keyProduct s x tile r e ((blocks s.T 32-1-j)*32+u),
       carry.2*gamma (tile.bh%s.H)^32+
         ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(64-r)*gamma (tile.bh%s.H)^u*valueProduct s x tile r v ((blocks s.T 32-1-j)*32+u)) := by
  unfold keyValueFuture
  simp_rw [dot_real,decayNat_real,blockDecay_real]
  simp only [real_zero,real_mul,real_cast,zero_add]
  apply Prod.ext
  · simp only [Prod.fst]
    congr 1
    apply Finset.sum_congr rfl
    intro u hu
    unfold keyProduct
    simp only [Finset.mul_sum,Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro f hf
    ring
  · simp only [Prod.snd]
    congr 1
    apply Finset.sum_congr rfl
    intro u hu
    unfold valueProduct
    simp only [Finset.mul_sum,Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro f hf
    ring

theorem keyValue_diagonal_step (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e v j : Nat) (carry : ℝ×ℝ) :
    keyValueDiagonal realOps p s x tile r e v j carry=
      (carry.1+(Real.sqrt (s.DK:ℝ))⁻¹*
         ∑ u∈Finset.range 32,if r≤j*32+u then gamma (tile.bh%s.H)^(j*32+u-r)*keyProduct s x tile r e (tile.chunk*64+j*32+u) else 0,
       carry.2+(Real.sqrt (s.DK:ℝ))⁻¹*
         ∑ u∈Finset.range 32,if r≤j*32+u then gamma (tile.bh%s.H)^(j*32+u-r)*valueProduct s x tile r v (tile.chunk*64+j*32+u) else 0) := by
  unfold keyValueDiagonal
  simp_rw [dot_real,real_cast,diagonalDecay_real]
  simp only [real_zero,real_mul,real_scale,zero_add]
  apply Prod.ext
  · simp only [Prod.fst]
    rw [Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro u hu
    split_ifs with hc
    · unfold keyProduct
      ring
    · ring
  · simp only [Prod.snd]
    rw [Finset.mul_sum]
    congr 1
    apply Finset.sum_congr rfl
    intro u hu
    split_ifs with hc
    · unfold valueProduct
      ring
    · ring

/-- Pair accumulator projection is proved by induction over actual loop steps. -/
theorem foldN_pair (f g : Nat → ℝ → ℝ) (n : Nat) (z : ℝ×ℝ) :
    foldN (fun j acc => (f j acc.1,g j acc.2)) n z=
      (foldN f n z.1,foldN g n z.2) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,ih]

theorem foldN_left_carry (g : ℝ) (x : Nat → ℝ) (n : Nat) (z : ℝ) :
    foldN (fun i acc => acc*g+x i) n z=g^n*z+leftScan g x n := by
  induction n with
  | zero => simp [foldN,leftScan]
  | succ n ih => rw [foldN,ih,leftScan,pow_succ];ring

theorem foldN_pair_zero (g : ℝ) (x y : Nat → ℝ) (n : Nat) :
    foldN (fun j (acc : ℝ×ℝ) => (acc.1*g+x j,acc.2*g+y j)) n (0,0)=
      (leftScan g x n,leftScan g y n) := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,leftScan,ih]

/-- Both decreasing-time accumulators are finite weighted sums of logical
products; dV retains its predecayed K factor at this stage. -/
theorem keyValue_future_sums (p : Plans) (s : Shape) (x : Inputs ℝ) (tile : Tile)
    (r e v N : Nat) :
    foldN (keyValueFuture realOps p s x tile r e v) N (0,0)=
      ((∑ j∈Finset.range N,(gamma (tile.bh%s.H)^32)^(N-1-j)*
          ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^u*keyProduct s x tile r e ((blocks s.T 32-1-j)*32+u)),
       (∑ j∈Finset.range N,(gamma (tile.bh%s.H)^32)^(N-1-j)*
          ∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(64-r)*gamma (tile.bh%s.H)^u*valueProduct s x tile r v ((blocks s.T 32-1-j)*32+u))) := by
  have eq : keyValueFuture realOps p s x tile r e v=
      (fun j carry =>
        (carry.1*gamma (tile.bh%s.H)^32+∑ u∈Finset.range 32,gamma (tile.bh%s.H)^u*keyProduct s x tile r e ((blocks s.T 32-1-j)*32+u),
         carry.2*gamma (tile.bh%s.H)^32+∑ u∈Finset.range 32,gamma (tile.bh%s.H)^(64-r)*gamma (tile.bh%s.H)^u*valueProduct s x tile r v ((blocks s.T 32-1-j)*32+u))) := by
    funext j carry
    exact keyValue_future_step p s x tile r e v j carry
  rw [eq,foldN_pair_zero,leftScan_sum,leftScan_sum]

#axiomsClean keyValue_future_step
#axiomsClean keyValue_diagonal_step
#axiomsClean keyValue_future_sums
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention

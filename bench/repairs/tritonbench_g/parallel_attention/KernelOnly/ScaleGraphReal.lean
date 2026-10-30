import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ScaleOps

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Math
open VeriTile.Bench.Repairs.ParallelAttention
variable {σ : ℝ}
noncomputable section
open scoped BigOperators

@[simp] theorem real_zero : (scaleOps σ).zero=(0:ℝ) := rfl
@[simp] theorem real_one : (scaleOps σ).one=(1:ℝ) := rfl
@[simp] theorem real_two : (scaleOps σ).two=(2:ℝ) := rfl
@[simp] theorem real_add (x y : ℝ) : (scaleOps σ).add x y=x+y := rfl
@[simp] theorem real_mul (x y : ℝ) : (scaleOps σ).mul x y=x*y := rfl
@[simp] theorem real_cast (x : ℝ) : (scaleOps σ).cast x=x := rfl
@[simp] theorem real_scale (K : Nat) (flag : Bool) : (scaleOps σ).scale K flag=σ := rfl

theorem dot_real (p : Plans) (site : List Nat) (n : Nat) (x y : Nat → ℝ) :
    dot (scaleOps σ) p site n x y=∑ e∈Finset.range n,x e*y e := by
  rw [dot, dot_scale_independent]
  simpa [VeriTile.Bench.Repairs.ParallelAttention.realOps] using (p.dotTree site n).real_sum n (p.dotComplete site n) x y 0

theorem total_real (p : Plans) (site : List Nat) (n : Nat) (x : Nat → ℝ) :
    total (scaleOps σ) p site n x=∑ e∈Finset.range n,x e := by
  rw [total, sum_scale_independent]
  exact (p.sumTree site n).real_sum n (p.sumComplete site n) x


/-- The guarded padded feature sum is exactly the logical score on valid rows. -/
theorem scaled_query_dot (p : Plans) (site : List Nat) (s : Shape) (x : TensorInputs ℝ)
    (bh t j : Nat) (ht : t<s.T) (hj : j<s.T) (fits : s.K ≤ s.BK) :
    dot (scaleOps σ) p site s.BK
      (fun e => (scaleOps σ).cast ((scaleOps σ).mul (load (scaleOps σ) s.T s.K x.q bh t e)
        ((scaleOps σ).scale s.K s.useScale)))
      (fun e => load (scaleOps σ) s.T s.K x.k bh j e)=
      score s.K (σ) (headInputs x bh) t j := by
  rw [dot_real (σ := σ)]
  simp only [real_cast (σ := σ),real_mul (σ := σ),real_scale (σ := σ)]
  have terms : (fun e => load (scaleOps σ) s.T s.K x.q bh t e * σ *
      load (scaleOps σ) s.T s.K x.k bh j e)=
      (fun e => if e<s.K then σ*(x.q bh t e*x.k bh j e) else 0) := by
    funext e
    by_cases he : e<s.K
    · simp only [load,ht,hj,he,and_self,if_true]
      ring
    · simp [load,ht,hj,he]
  rw [terms,sum_padded s.K s.BK fits]
  simp only [score,headInputs,← Finset.mul_sum]

/-- The unscaled query/key orientation in DK/DV computes the same Real score;
its different source scale placement is retained in the FP graph. -/
theorem key_query_dot (p : Plans) (site : List Nat) (s : Shape) (x : TensorInputs ℝ)
    (bh t j : Nat) (ht : t<s.T) (hj : j<s.T) (fits : s.K ≤ s.BK) :
    (scaleOps σ).mul (dot (scaleOps σ) p site s.BK
      (fun e => load (scaleOps σ) s.T s.K x.k bh j e)
      (fun e => load (scaleOps σ) s.T s.K x.q bh t e)) ((scaleOps σ).scale s.K s.useScale)=
      score s.K (σ) (headInputs x bh) t j := by
  rw [real_mul (σ := σ),real_scale (σ := σ),dot_real (σ := σ)]
  have terms : (fun e => load (scaleOps σ) s.T s.K x.k bh j e * load (scaleOps σ) s.T s.K x.q bh t e)=
      (fun e => if e<s.K then x.q bh t e*x.k bh j e else 0) := by
    funext e
    by_cases he : e<s.K
    · simp only [load,ht,hj,he,and_self,if_true]
      ring
    · simp [load,ht,hj,he]
  rw [terms,sum_padded s.K s.BK fits]
  simp only [score,headInputs]
  ring

def partialCotangent (s : Shape) (x : TensorInputs ℝ) (bh shard t j : Nat) : ℝ :=
  (∑ v∈Finset.range s.BV,
    if shard*s.BV+v<s.V then x.do_ bh t (shard*s.BV+v)*x.v bh j (shard*s.BV+v) else 0)+
    (if shard=0 then x.dz bh t else 0)

theorem cotangent_shards (s : Shape) (x : TensorInputs ℝ) (bh t j : Nat) (hbv : 0<s.BV)
    (hv : 0<s.V) :
    (∑ shard∈Finset.range (blocks s.V s.BV),partialCotangent s x bh shard t j)=
      scoreCotangent s.V (headInputs x bh) (headCotangent x bh) t j := by
  have nonempty : 0<blocks s.V s.BV := by
    have cover := blocks_cover s.V s.BV hbv
    by_contra h
    have hz : blocks s.V s.BV=0 := by omega
    rw [hz] at cover
    omega
  unfold partialCotangent
  rw [Finset.sum_add_distrib,tiled_partition s.V s.BV hbv (fun p => x.do_ bh t p * x.v bh j p)]
  simp only [scoreCotangent,headInputs,headCotangent]
  congr 1
  simp [Finset.sum_ite_eq',nonempty]

theorem foldN_add (f : Nat → ℝ) (n : Nat) (z : ℝ) :
    foldN (fun i carry => carry+f i) n z=z+∑ i∈Finset.range n,f i := by
  induction n with
  | zero => simp [foldN]
  | succ n ih => rw [foldN,ih,Finset.sum_range_succ];ring

theorem foldN_pair_add (f g : Nat → ℝ) (n : Nat) (z : ℝ×ℝ) :
    foldN (fun i carry => (carry.1+f i,carry.2+g i)) n z=
      (z.1+∑ i∈Finset.range n,f i,z.2+∑ i∈Finset.range n,g i) := by
  induction n with
  | zero => simp [foldN]
  | succ n ih =>
    rw [foldN,ih,Finset.sum_range_succ,Finset.sum_range_succ]
    apply Prod.ext <;> dsimp <;> ring

#axiomsClean scaled_query_dot
#axiomsClean key_query_dot
#axiomsClean cotangent_shards
#axiomsClean foldN_add
#axiomsClean foldN_pair_add
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Math

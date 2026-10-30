import bench.tritonbench_g.rms_matmul_rbe.Proofs.Core
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.Repairs.RmsMatmulRbe
open Int8DequantMatmul (rsum rsum_congr rsum_swap rsum_blocks rsum_padding)
noncomputable section
set_option maxHeartbeats 5000000
set_option linter.unusedSimpArgs false

/-- Ideal interpretation erases storage casts and uses mathematical operations.
It is separate from the native word interpretation; no exact-FP-to-Real premise. -/
def realOps : Ops ℝ where
  zero := 0
  one := 1
  add := (·+·)
  mul := (·*·)
  mulFTZ := (·*·)
  divide := (·/·)
  rsqrt := fun x => 1/Real.sqrt x
  neg := Neg.neg
  fma := fun x y z => x*y+z
  ofInt := Int.cast
  pow := Real.rpow
  cos := Real.cos
  sin := Real.sin
  mma8 := fun x y old => old+(List.ofFn (fun i => x i*y i)).sum

theorem finRange_rsum (f : Nat → ℝ) (n : Nat) :
    ((List.finRange n).map (fun i => f i.val)).sum=rsum f n := by
  induction n with
  | zero => simp [rsum]
  | succ n ih => simpa [List.finRange_succ_last,List.map_append,List.map_map,Function.comp_def,rsum] using congrArg (fun v => v+f n) ih

theorem Tree.real (t : Tree) (v : Fin 64 → ℝ) :
    t.eval (·+·) v=(t.leaves.map v).sum := by
  induction t with
  | leaf k => simp [Tree.eval,Tree.leaves]
  | node a b ia ib => simp [Tree.eval,Tree.leaves,ia,ib]
theorem Reduction.real (r : Reduction) (v : Nat → ℝ) :
    r.tree.eval (·+·) (fun k => v k.val)=rsum v 64 := by
  rw [Tree.real,(r.complete.map (fun k => v k.val)).sum_eq,finRange_rsum]

theorem instructions_real (p : Parameters ℝ) (x : RowInput ℝ) (n t : Nat)
    (ops : List (Fin 8 → Fin 64)) (old : ℝ) :
    ops.foldl (fun v lanes => instruction realOps p x n t lanes v) old=
      old+((ops.flatMap List.ofFn).map
        (fun k => scaled realOps p x t k * masked realOps p (x.weight n) t k)).sum := by
  induction ops generalizing old with
  | nil => simp
  | cons lanes ops ih =>
    rw [List.foldl_cons,ih]
    simp only [instruction,realOps,List.flatMap_cons,List.map_append,List.sum_append,List.map_ofFn,Function.comp_def]
    ring
theorem dotBlock_real (p : Parameters ℝ) (x : RowInput ℝ) (plan : DotPlan)
    (n t : Nat) (old : ℝ) :
    dotBlock realOps p x plan n t old=old+rsum
      (fun k => if t*64+k<p.K then (x.x (t*64+k)*x.rms (t*64+k))*x.weight n (t*64+k) else 0) 64 := by
  unfold dotBlock
  rw [instructions_real,(plan.complete.map
    (fun k => scaled realOps p x t k * masked realOps p (x.weight n) t k)).sum_eq]
  have h (k : Fin 64) : scaled realOps p x t k * masked realOps p (x.weight n) t k=
      if t*64+k.val<p.K then (x.x (t*64+k.val)*x.rms (t*64+k.val))*x.weight n (t*64+k.val) else 0 := by
    by_cases hk : t*64+k.val<p.K <;> simp [scaled,masked,hk,realOps]
  simp_rw [h]
  rw [finRange_rsum (fun k => if t*64+k<p.K then (x.x (t*64+k)*x.rms (t*64+k))*x.weight n (t*64+k) else 0) 64]
theorem accumulation_real (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → DotPlan) (n count : Nat) :
    accumulation realOps p x plans n count=rsum (fun t => rsum
      (fun k => if t*64+k<p.K then (x.x (t*64+k)*x.rms (t*64+k))*x.weight n (t*64+k) else 0) 64) count := by
  induction count with
  | zero => rfl
  | succ t ih => rw [accumulation,dotBlock_real,ih]; rfl
theorem k_partition (K : Nat) (f : Nat → ℝ) :
    rsum (fun t => rsum (fun k => if t*64+k<K then f (t*64+k) else 0) 64) (steps K)=rsum f K := by
  rw [rsum_blocks (fun k => if k<K then f k else 0) 64,rsum_padding,Nat.min_eq_left]
  have h := IntScaledMatmul.Revision001.LogicalSum.ceil_cover K 64 (by decide)
  simpa [steps] using h
theorem projection_sum (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → DotPlan) (n : Nat) :
    accumulation realOps p x plans n (steps p.K)=rsum
      (fun k => (x.x k*x.rms k)*x.weight n k) p.K := by
  rw [accumulation_real]
  exact k_partition p.K (fun k => (x.x k*x.rms k)*x.weight n k)

theorem moment_real (p : Parameters ℝ) (x : RowInput ℝ) (l : Fin 64) (count : Nat) :
    moment realOps p x l count=rsum
      (fun t => if t*64+l.val<p.K then x.x (t*64+l.val)^2 else 0) count := by
  induction count with
  | zero => rfl
  | succ t ih =>
    rw [moment,ih]
    by_cases h : t*64+l.val<p.K <;> simp [rsum,square,masked,h,realOps,pow_two]
theorem moment_sum (p : Parameters ℝ) (x : RowInput ℝ) (reduction : Reduction) :
    reduction.tree.eval realOps.add (fun l => moment realOps p x l (steps p.K))=
      rsum (fun k => x.x k^2) p.K := by
  simp_rw [moment_real]
  change reduction.tree.eval (·+·) _=_
  rw [Reduction.real reduction (fun l => rsum (fun t => if t*64+l<p.K then x.x (t*64+l)^2 else 0) (steps p.K)),rsum_swap]
  exact k_partition p.K (fun k => x.x k^2)

def realNorm (p : Parameters ℝ) (x : RowInput ℝ) :=
  1/Real.sqrt (rsum (fun k => x.x k^2) p.K/(p.K:ℝ)+p.epsilon)
def realProjection (p : Parameters ℝ) (x : RowInput ℝ) (n : Nat) :=
  (rsum (fun k => x.x k*x.rms k*x.weight n k) p.K)*realNorm p x
/-- Independent mathematical target: RMS-normalized GEMM, adjacent pairs
within each head for rotary=True, unchanged projection for rotary=False. -/
def realTarget (p : Parameters ℝ) (x : RowInput ℝ) (rotary : Bool) (row n : Nat) :=
  if rotary then
    let even := n-n%2
    let phase := ((p.start:ℝ)+(row:ℝ))/
      Real.rpow p.theta ((((even%p.headDim):Nat):ℝ)/(p.headDim:ℝ))
    if n%2=0 then realProjection p x even*Real.cos phase-realProjection p x (even+1)*Real.sin phase
    else realProjection p x even*Real.sin phase+realProjection p x (even+1)*Real.cos phase
  else realProjection p x n
theorem norm_real (p : Parameters ℝ) (x : RowInput ℝ) (reduction : Reduction) :
    norm realOps p x reduction=realNorm p x := by
  unfold norm
  rw [moment_sum]
  simp [realNorm,realOps]
theorem projection_real (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (n : Nat) :
    projection realOps p x plans reduction n=realProjection p x n := by
  rw [projection,projection_sum,norm_real]
  rfl
theorem value_real (p : Parameters ℝ) (x : RowInput ℝ)
    (plans : Nat → Nat → DotPlan) (reduction : Reduction) (rotary : Bool) (row n : Nat) :
    value realOps p x plans reduction rotary row n=realTarget p x rotary row n := by
  have pair : n/2*2=n-n%2 := by omega
  simp only [value]
  simp only [projection_real,pair]
  simp only [rotate,angle,realOps,Int.cast_add,Int.cast_natCast,realTarget]
  split_ifs <;> ring

#axiomsClean dotBlock_real
#axiomsClean accumulation_real
#axiomsClean k_partition
#axiomsClean moment_sum
#axiomsClean value_real
end
end VeriTile.Bench.Repairs.RmsMatmulRbe

import bench.repairs.tritonbench_g.parallel_attention.Proofs.Lowering
import bench.repairs.tritonbench_g.parallel_attention.Proofs.GraphReal

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Resolved scalar instruction shapes. These are finite syntax, not numerical
callbacks, and retain compiler contraction and operand order. -/
theorem normEval_real (p : DotPlan) (raw : Nat → ℝ) (masked : Nat → Bool) :
    normEval realOps raw masked p=
      p.eval realOps (fun i => if masked i then 0 else raw i)
        (fun i => if masked i then 0 else raw i) 0 := by
  induction p with
  | zero => rfl
  | acc => rfl
  | product i => cases h : masked i <;> simp [normEval,DotPlan.eval,realOps,h]
  | add l r hl hr => simp only [normEval,DotPlan.eval,hl,hr]
  | fma i t ht => simp only [normEval,DotPlan.eval,ht,real_zero]

theorem nativeNorm_real (c : CompilerLayout) (site : List Nat) (n : Nat)
    (raw : Nat → ℝ) (masked : Nat → Bool) :
    nativeNorm realOps c site n raw masked=
      ∑ i∈Finset.range n,if masked i then 0 else raw i*raw i := by
  rw [nativeNorm,normEval_real,DotPlan.real_leaves,c.normNoCarry]
  simp only [Nat.cast_zero,zero_mul,zero_add]
  have same (i : Nat) : (if masked i then (0:ℝ) else raw i)*(if masked i then 0 else raw i)=
      (if masked i then 0 else raw i*raw i) := by
    cases h : masked i <;> simp [h]
  simpa [same] using List.Perm.sum_eq ((c.normComplete site n).map
    (fun i => (if masked i then 0 else raw i)*(if masked i then 0 else raw i)))

theorem nativeCoefficient_real (c : CompilerLayout) (site : List Nat)
    (valid : Bool) (twice : ℝ → ℝ) (ds score : ℝ) :
    nativeCoefficient realOps c site valid twice ds score=
      if valid then twice ds*score else 0 := by
  cases valid <;> cases ho : c.scalar.coefficientOrder site <;>
    cases hm : c.scalar.coefficientMask site <;>
    simp [nativeCoefficient,ho,hm,ProductOrder.eval,realOps,mul_comm]

theorem liveDot_real (p : Plans) (site : List Nat) (n : Nat) (x y : Nat → ℝ) (carry : ℝ) :
    liveDot realOps p site n x y carry=carry+dot realOps p site n x y := by
  rw [liveDot,(p.dotTree site n).real_sum n (p.dotComplete site n),dot_real]

theorem nativeScale_real (c : CompilerLayout) (s : Shape) (site : List Nat) (x : ℝ) :
    nativeScale realOps c s site x=x*selectedScale s.K s.useScale := by
  cases h : s.useScale <;> cases hf : c.scalar.scale site <;> simp [nativeScale,h,hf,ScaleForm.eval,selectedScale,mul_comm]

theorem nativeTwice_real (c : CompilerLayout) (site : List Nat) (x : ℝ) :
    nativeTwice realOps c site x=2*x := by
  unfold nativeTwice
  cases c.scalar.twice site <;> simp [TwiceForm.eval] <;> ring

theorem nativeScaleTwice_real (c : CompilerLayout) (s : Shape) (site : List Nat) (x : ℝ) :
    nativeScaleTwice realOps c s site x=2*(x*selectedScale s.K s.useScale) := by
  cases h : s.useScale
  · simp [nativeScaleTwice,h,nativeTwice_real,selectedScale]
  · cases hf : c.scalar.scaleTwice site <;> simp [nativeScaleTwice,h,hf,ScaleTwiceForm.eval,realOps,selectedScale] <;> ring

theorem nativeScaledSum_real (c : CompilerLayout) (s : Shape) (site : List Nat) (x z : ℝ) :
    nativeScaledSum realOps c s site x z=(x+z)*selectedScale s.K true := by
  unfold nativeScaledSum
  cases c.scalar.scaledSum site <;> simp [ScaledSumForm.eval,realOps,selectedScale] <;> ring


end
end VeriTile.Bench.Repairs.ParallelAttention

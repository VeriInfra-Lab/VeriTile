import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.QueryGraph
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.ScaleOps

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttention
noncomputable section

/-- Fixed-arity scalar ports only. Tensor operations are constructed below. -/
structure Scalars (α : Type) where
  literal : ℝ → α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  cast : α → α
  div : α → α → α
  neg : α → α

def Scalars.ops (a : Scalars α) (scale : ℝ) : Ops α where
  zero := a.literal 0
  one := a.literal 1
  two := a.literal 2
  add := a.add
  mul := a.mul
  fma := a.fma
  cast := a.cast
  div := a.div
  neg := a.neg
  scale := fun _ _ => a.literal scale

def serialSum : Nat → SumPlan
  | 0 => .zero
  | n+1 => .add (serialSum n) (.input n)

theorem serialSum_complete (n : Nat) : (serialSum n).Covers n := by
  induction n with
  | zero => exact .refl _
  | succ n ih => simpa [serialSum, SumPlan.Covers, SumPlan.leaves, List.range_succ]
      using ih.append (List.Perm.refl [n])

/-- A fixed source primitive profile; no caller-supplied reduction plan. Native
compiler contraction and routing are separately represented by the native DAG. -/
def sourcePlans : Plans where
  dotTree := fun _ => DotPlan.linear
  dotComplete := fun _ => DotPlan.linear_complete
  sumTree := fun _ => serialSum
  sumComplete := fun _ => serialSum_complete

def scalarDot (a : Scalars α) : (batch : TileShape) → {m k n : Nat} →
    Values α .real (batch++[m,k]) → Values α .real (batch++[k,n]) →
    Values α .real (batch++[m,n])
  | [], _, k, _, x, y => fun i => (DotPlan.linear k).eval (a.ops 0)
      (fun j => if h : j<k then x ⟨i.1,⟨j,h⟩,()⟩ else a.literal 0)
      (fun j => if h : j<k then y ⟨⟨j,h⟩,i.2.1,()⟩ else a.literal 0) (a.literal 0)
  | _::rest, _, _, _, x, y => fun i => scalarDot a rest (fun j => x ⟨i.1,j⟩) (fun j => y ⟨i.1,j⟩) i.2

def scalarDotPrimitive (a : Scalars α) : DotPrimitive α where
  apply := fun _ _ {batch} {_} {_} {_} x y => scalarDot a batch x y

def scalarSum (a : Scalars α) {shape : TileShape} (ax : Fin shape.length)
    (keep : Bool) (xs : Values α .real shape) : Values α .real (TileShape.reduceShape shape ax keep) := by
  cases keep
  · exact fun out => (serialSum (TileShape.axisDim shape ax)).eval (a.ops 0)
      (fun j => if h : j<TileShape.axisDim shape ax then
        xs (TileShape.insertAxisIndex shape ax out ⟨j,h⟩) else a.literal 0)
  · exact fun out => (serialSum (TileShape.axisDim shape ax)).eval (a.ops 0)
      (fun j => if h : j<TileShape.axisDim shape ax then
        xs (TileShape.replaceAxisIndex shape ax out ⟨j,h⟩) else a.literal 0)

def scalarAlgebra (a : Scalars α) : Algebra α where
  literal := fun _ _ => a.literal
  negInf := a.literal 0
  binary := fun _ _ op x y => match op with
    | .add => a.add x y | .mul => a.mul x y | _ => a.literal 0
  unary := fun _ _ _ => a.literal 0
  cast := fun _ _ _ => a.cast
  fromNat := fun _ n => a.literal n
  fromInt := fun _ n => a.literal n
  fp32Bits := fun _ => a.literal 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => a.literal 0
  reduceSum := fun _ {_} ax keep xs => scalarSum a ax keep xs

theorem dotPlan_scale (a : Scalars α) (scale : ℝ) (p : DotPlan) (x y : Nat → α) (z : α) :
    p.eval (a.ops 0) x y z = p.eval (a.ops scale) x y z := by
  induction p with
  | zero | acc | product => rfl
  | add l r hl hr => exact congrArg₂ a.add hl hr
  | fma i p ih => exact congrArg (a.fma (x i) (y i)) ih

theorem sumPlan_scale (a : Scalars α) (scale : ℝ) (p : SumPlan) (x : Nat → α) :
    p.eval (a.ops 0) x = p.eval (a.ops scale) x := by
  induction p with
  | zero | input => rfl
  | add l r hl hr => exact congrArg₂ a.add hl hr

theorem linear_dot_congr (a : Ops α) (n : Nat) (x y u v : Nat → α) (z : α)
    (hx : ∀ j, j<n → x j=u j) (hy : ∀ j, j<n → y j=v j) :
    (DotPlan.linear n).eval a x y z = (DotPlan.linear n).eval a u v z := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [DotPlan.linear, DotPlan.eval]
    rw [hx n (by omega), hy n (by omega), ih (by intro j hj; exact hx j (by omega))
      (by intro j hj; exact hy j (by omega))]

theorem serial_sum_congr (a : Ops α) (n : Nat) (x y : Nat → α)
    (hx : ∀ j, j<n → x j=y j) :
    (serialSum n).eval a x = (serialSum n).eval a y := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [serialSum, SumPlan.eval]
    rw [hx n (by omega), ih (by intro j hj; exact hx j (by omega))]

#axiomsClean serialSum_complete
#axiomsClean dotPlan_scale
#axiomsClean sumPlan_scale
#axiomsClean linear_dot_congr
#axiomsClean serial_sum_congr
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel

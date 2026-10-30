import Mathlib
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Every arithmetic port is one scalar operation. There is no dot,
reduction, output tensor, successful-run, or final-memory callback. -/
structure Ops (α : Type) where
  zero : α
  one : α
  two : α
  add : α → α → α
  mul : α → α → α
  fma : α → α → α → α
  div : α → α → α
  neg : α → α
  cast : α → α
  scale : Nat → Bool → α

def realOps : Ops ℝ where
  zero := 0
  one := 1
  two := 2
  add := (·+·)
  mul := (·*·)
  fma := fun x y z => x*y+z
  div := (·/·)
  neg := Neg.neg
  cast := id
  scale := fun n useScale => if useScale then (Real.sqrt (n:ℝ))⁻¹ else 1

/-- A scalar FP32 dot expression, including the actual live accumulator.
SIMT multiply/add and FMA trees are explicit; input coverage is combinatorial. -/
inductive DotPlan where
  | zero
  | acc
  | product (index : Nat)
  | add (left right : DotPlan)
  | fma (index : Nat) (tail : DotPlan)
  deriving Repr, DecidableEq

def DotPlan.leaves : DotPlan → List Nat
  | .zero | .acc => []
  | .product i => [i]
  | .add l r => l.leaves++r.leaves
  | .fma i t => i::t.leaves
def DotPlan.carries : DotPlan → Nat
  | .zero | .product _ => 0
  | .acc => 1
  | .add l r => l.carries+r.carries
  | .fma _ t => t.carries
def DotPlan.eval (a : Ops α) (x y : Nat → α) (carry : α) : DotPlan → α
  | .zero => a.zero
  | .acc => carry
  | .product i => a.mul (x i) (y i)
  | .add l r => a.add (l.eval a x y carry) (r.eval a x y carry)
  | .fma i t => a.fma (x i) (y i) (t.eval a x y carry)
def DotPlan.Covers (p : DotPlan) (n : Nat) : Prop :=
  p.leaves.Perm (List.range n) ∧ p.carries=1

theorem DotPlan.real_leaves (p : DotPlan) (x y : Nat → ℝ) (carry : ℝ) :
    p.eval realOps x y carry=(p.carries : ℝ)*carry+(p.leaves.map (fun i => x i*y i)).sum := by
  induction p with
  | zero => simp [DotPlan.eval,DotPlan.carries,DotPlan.leaves,realOps]
  | acc => simp [DotPlan.eval,DotPlan.carries,DotPlan.leaves,realOps]
  | product i => simp [DotPlan.eval,DotPlan.carries,DotPlan.leaves,realOps]
  | add l r hl hr =>
    change l.eval realOps x y carry+r.eval realOps x y carry=_
    rw [hl,hr]
    simp only [DotPlan.carries,DotPlan.leaves,List.map_append,List.sum_append,Nat.cast_add]
    ring
  | fma i t ht =>
    change x i*y i+t.eval realOps x y carry=_
    rw [ht]
    simp only [DotPlan.carries,DotPlan.leaves,List.map_cons,List.sum_cons]
    ring

theorem DotPlan.real_sum (p : DotPlan) (n : Nat) (complete : p.Covers n)
    (x y : Nat → ℝ) (carry : ℝ) :
    p.eval realOps x y carry=carry+∑ i∈Finset.range n,x i*y i := by
  rw [p.real_leaves,complete.2]
  simp only [Nat.cast_one,one_mul]
  congr 1
  simpa using List.Perm.sum_eq (complete.1.map (fun i => x i*y i))

def DotPlan.linear : Nat → DotPlan
  | 0 => .acc
  | n+1 => .fma n (linear n)
theorem DotPlan.linear_complete (n : Nat) : (DotPlan.linear n).Covers n := by
  induction n with
  | zero => exact ⟨List.Perm.refl _,rfl⟩
  | succ n ih =>
    constructor
    · simpa [linear,leaves,List.range_succ] using
        (List.Perm.cons n ih.1).trans (List.perm_append_comm : ([n]++List.range n).Perm (List.range n++[n]))
    · simpa [linear,carries] using ih.2

/-- Host sum(0) is an explicit finite addition tree, including zero seeds.
Its actual order is supplied by an ordinary Torch lowering plan. -/
inductive SumPlan where
  | zero
  | input (index : Nat)
  | add (left right : SumPlan)
  deriving Repr, DecidableEq
def SumPlan.leaves : SumPlan → List Nat
  | .zero => []
  | .input i => [i]
  | .add l r => l.leaves++r.leaves
def SumPlan.eval (a : Ops α) (x : Nat → α) : SumPlan → α
  | .zero => a.zero
  | .input i => x i
  | .add l r => a.add (l.eval a x) (r.eval a x)
def SumPlan.Covers (p : SumPlan) (n : Nat) : Prop := p.leaves.Perm (List.range n)
theorem SumPlan.real_leaves (p : SumPlan) (x : Nat → ℝ) :
    p.eval realOps x=(p.leaves.map x).sum := by
  induction p with
  | zero => rfl
  | input i => simp [SumPlan.eval,SumPlan.leaves]
  | add l r hl hr =>
    change l.eval realOps x+r.eval realOps x=_
    rw [hl,hr]
    simp [SumPlan.leaves]
theorem SumPlan.real_sum (p : SumPlan) (n : Nat) (complete : p.Covers n) (x : Nat → ℝ) :
    p.eval realOps x=∑ i∈Finset.range n,x i := by
  rw [p.real_leaves]
  simpa using List.Perm.sum_eq (complete.map x)

def foldN (step : Nat → β → β) : Nat → β → β
  | 0,z => z
  | n+1,z => step n (foldN step n z)
theorem foldN_congr (f g : Nat → β → β) (same : ∀ i z,f i z=g i z) (n : Nat) (z : β) :
    foldN f n z=foldN g n z := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,ih,same]

#axiomsClean DotPlan.real_sum
#axiomsClean DotPlan.linear_complete
#axiomsClean SumPlan.real_sum
end
end VeriTile.Bench.Repairs.ParallelAttention

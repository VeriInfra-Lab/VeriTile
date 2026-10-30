import bench.repairs.tritonbench_g.fused_recurrent_retention.Kernels
import Mathlib
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.ScalarOps

namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
open scoped BigOperators

theorem foldN_congr (f g : Nat → β → β) (h : ∀ i z,f i z=g i z) (n : Nat) (z : β) :
    foldN f n z=foldN g n z := by
  induction n with
  | zero => rfl
  | succ n ih => simp only [foldN,ih,h]

/- A numerical-free scalar expression plan. A node is one actual add, mul,
or FMA. In particular, a multiply followed by a reduction may contract to
FMA; the plan must retain that contraction and operand order. -/
theorem DotPlan.real_leaves (p : DotPlan) (x y : Nat → ℝ) :
    p.eval realArithmetic x y=(p.leaves.map (fun i => x i*y i)).sum := by
  induction p with
  | product i => simp [DotPlan.eval,DotPlan.leaves,realArithmetic]
  | add l r hl hr =>
    change l.eval realArithmetic x y+r.eval realArithmetic x y=_
    rw [hl,hr]
    simp [DotPlan.leaves]
  | fma i tail ih =>
    change x i*y i+tail.eval realArithmetic x y=_
    rw [ih]
    simp [DotPlan.leaves]

theorem DotPlan.real_sum (p : DotPlan) (n : Nat) (h : p.Covers n) (x y : Nat → ℝ) :
    p.eval realArithmetic x y=∑ i∈Finset.range n,x i*y i := by
  rw [p.real_leaves]
  have e := List.Perm.sum_eq (h.map (fun i => x i*y i))
  simpa using e

/- Torch's sum(0) is a finite addition plan, including its zero-initialized
thread accumulators and later block/global combines. No result callback is
present; the ordinary runtime binding supplies the actual plan. -/
theorem SumPlan.real_leaves (p : SumPlan) (x : Nat → ℝ) :
    p.eval realArithmetic x=(p.leaves.map x).sum := by
  induction p with
  | zero => rfl
  | input i => simp [SumPlan.eval,SumPlan.leaves]
  | add l r hl hr =>
    change l.eval realArithmetic x+r.eval realArithmetic x=_
    rw [hl,hr]
    simp [SumPlan.leaves]

theorem SumPlan.real_sum (p : SumPlan) (n : Nat) (h : p.Covers n) (x : Nat → ℝ) :
    p.eval realArithmetic x=∑ i∈Finset.range n,x i := by
  rw [p.real_leaves]
  have e := List.Perm.sum_eq (h.map x)
  simpa using e

#axiomsClean DotPlan.real_sum
#axiomsClean SumPlan.real_sum
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention

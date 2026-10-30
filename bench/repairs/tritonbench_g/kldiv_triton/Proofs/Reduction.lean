import VeriTile.Triton.Float.TermModel
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Math

namespace VeriTile.Bench.Repairs.KldivTriton.HostReduction
open VeriTile.Triton.FP.Equational
open scoped BigOperators
noncomputable section
set_option autoImplicit false

/-- Ordinary Torch's chosen addition plan is retained, including all padding
and operand order. No reassociation or whole-sum numerical law is assumed. -/
def eval {α : Type} {n : Nat} (add : α → α → α) (zero : α) (values : Fin n → α) : ReductionTree n → α
  | .input i => values i
  | .zero => zero
  | .add left right => add (eval add zero values left) (eval add zero values right)

theorem eval_congr {α : Type} {n : Nat} (add : α → α → α) (zero : α)
    (xs ys : Fin n → α) (same : ∀ i,xs i=ys i) (tree : ReductionTree n) :
    eval add zero xs tree=eval add zero ys tree := by
  have h : xs=ys := funext same
  rw [h]

theorem real_leaves {n : Nat} (values : Fin n → ℝ) (tree : ReductionTree n) :
    eval (·+·) 0 values tree=(tree.leaves.map (fun i => i.elim 0 values)).sum := by
  induction tree with
  | input i => simp [eval,ReductionTree.leaves]
  | zero => simp [eval,ReductionTree.leaves]
  | add left right il ir => simp [eval,ReductionTree.leaves,il,ir,List.sum_append]

theorem real_plan {n : Nat} (values : Fin n → ℝ) (plan : ReductionPlan n) :
    eval (·+·) 0 values plan.tree=∑ i,values i := by
  rw [real_leaves]
  have perm := plan.valid.map (fun i => i.elim (0:ℝ) values)
  rw [List.Perm.sum_eq perm]
  simp only [List.map_append,List.map_map,Function.comp_def,Option.elim_some,List.map_replicate,Option.elim_none,List.sum_append]
  simpa using (Fin.sum_univ_def values).symm

#axiomsClean eval_congr
#axiomsClean real_plan
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivTriton.HostReduction

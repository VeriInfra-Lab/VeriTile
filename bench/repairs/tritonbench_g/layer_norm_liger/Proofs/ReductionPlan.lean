import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Grid
namespace VeriTile.Bench.Repairs.LayerNormLiger.ReductionPlan
open VeriTile.Triton FP.Structural
noncomputable section

/-- An explicit scalar-addition tree. Zero nodes represent the backend's
explicit initial accumulators, not an assumed floating additive identity. -/
inductive Tree (n : Nat) where
  | zero
  | leaf (index : Fin n)
  | add (left right : Tree n)

def Tree.leaves {n : Nat} : Tree n → List (Fin n)
  | .zero => []
  | .leaf i => [i]
  | .add l r => l.leaves ++ r.leaves

def Tree.eval {α : Type} {n : Nat} (zero : α) (add : α → α → α)
    (input : Fin n → α) : Tree n → α
  | .zero => zero
  | .leaf i => input i
  | .add l r => add (l.eval zero add input) (r.eval zero add input)

/-- Ordinary reduction lowering must bind one complete leaf-permutation tree;
no row, reduction result, or error bound is supplied by this contract. -/
structure Plan (n : Nat) where
  tree : Tree n
  coverage : tree.leaves.Perm (List.finRange n)

theorem Tree.real_sum {n : Nat} (t : Tree n) (input : Fin n → ℝ) :
    t.eval 0 (·+·) input=(t.leaves.map input).sum := by
  induction t with
  | zero => rfl
  | leaf i => simp [eval,leaves]
  | add l r hl hr => simp [eval,leaves,hl,hr,List.sum_append]

theorem Plan.real_sum {n : Nat} (p : Plan n) (input : Fin n → ℝ) :
    p.tree.eval 0 (·+·) input=((List.finRange n).map input).sum := by
  rw [Tree.real_sum]
  exact (p.coverage.map input).sum_eq

def Plan.bits {n : Nat} (p : Plan n) (cfg : FP.Config) (input : Fin n → FP.Value .fp32) :=
  p.tree.eval (FP.Value.zero .fp32) (FP.add cfg) input

#axiomsClean Tree.real_sum
#axiomsClean Plan.real_sum
end
end VeriTile.Bench.Repairs.LayerNormLiger.ReductionPlan

import bench.repairs.tritonbench_g.cross_entropy2.Proofs.Math

namespace VeriTile.Bench.Repairs.CrossEntropy2.Reduction
noncomputable section

/-- Explicit ordered binary reduction. Leaves are data indices, not a
callback supplying a complete numerical reduction result. -/
inductive Tree where
  | leaf (index : Nat)
  | seed
  | node (left right : Tree)
  deriving DecidableEq

def Tree.leaves : Tree → List Nat
  | .leaf i => [i]
  | .seed => []
  | .node a b => a.leaves++b.leaves

def Tree.run {α : Type} (op : α → α → α) (seed : α) (input : Nat → α) : Tree → α
  | .leaf i => input i
  | .seed => seed
  | .node a b => op (a.run op seed input) (b.run op seed input)

structure Plan (width : Nat) where
  tree : Tree
  covers : tree.leaves.Perm (List.range width)
  positive : 0<width

theorem tree_local_equiv {α β : Type} (left : α → α → α) (right : β → β → β)
    (relation : α → β → Prop) (step : ∀ a b c d, relation a c → relation b d → relation (left a b) (right c d))
    (seedX : α) (seedY : β) (seeds : relation seedX seedY)
    (x : Nat → α) (y : Nat → β) (leaves : ∀ i, relation (x i) (y i)) (t : Tree) :
    relation (t.run left seedX x) (t.run right seedY y) := by
  induction t with
  | leaf i => exact leaves i
  | seed => exact seeds
  | node a b ia ib => exact step _ _ _ _ ia ib

theorem real_tree (t : Tree) (x : Nat → ℝ) :
    t.run (·+·) 0 x=(t.leaves.map x).sum := by
  induction t with
  | leaf i => simp [Tree.run,Tree.leaves]
  | seed => rfl
  | node a b ia ib => simp [Tree.run,Tree.leaves,ia,ib]

/-- Full reduction correctness is proved by enumerating all leaves. -/
theorem real_plan {width : Nat} (p : Plan width) (x : Nat → ℝ) :
    p.tree.run (·+·) 0 x=((List.range width).map x).sum := by
  rw [real_tree]
  exact (p.covers.map x).sum_eq

theorem positive_width {width : Nat} (p : Plan width) : 0<width := p.positive

#axiomsClean tree_local_equiv
#axiomsClean real_plan
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Reduction

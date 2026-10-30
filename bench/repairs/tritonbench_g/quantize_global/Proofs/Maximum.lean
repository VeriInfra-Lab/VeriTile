import bench.repairs.tritonbench_g.quantize_global.Proofs.Model
namespace QuantizeGlobalRepair

/-- Explicit scalar reduction tree. The ordinary Torch reduction contract
supplies a tree whose leaves cover the logical tensor, not a maximum oracle. -/
inductive MaxTree where
  | empty
  | leaf (index : Nat)
  | node (left right : MaxTree)

def MaxTree.leaves : MaxTree → List Nat
  | .empty => [] | .leaf i => [i] | .node l r => l.leaves ++ r.leaves
def MaxTree.eval (zero : α) (absolute : α → α) (maximum : α → α → α)
    (xs : Nat → α) : MaxTree → α
  | .empty => zero
  | .leaf i => absolute (xs i)
  | .node l r => maximum (l.eval zero absolute maximum xs) (r.eval zero absolute maximum xs)
def MaxTree.Covers (tree : MaxTree) (n : Nat) : Prop := ∀ i, i ∈ tree.leaves ↔ i < n

end QuantizeGlobalRepair

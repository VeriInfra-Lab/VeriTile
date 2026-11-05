import VeriTile.Triton

namespace VeriTile.Bench.Repairs.FusedLayernormTriton.Kernels
open VeriTile.Triton
noncomputable section

/-- Source-selected binary helper tree; its order is numerical configuration. -/
inductive ReductionTree (N : Nat) where
  | leaf : Fin N → ReductionTree N
  | node : ReductionTree N → ReductionTree N → ReductionTree N

def ReductionTree.leaves {N : Nat} : ReductionTree N → List (Fin N)
  | .leaf i => [i]
  | .node l r => l.leaves ++ r.leaves

def ReductionTree.Complete {N : Nat} (t : ReductionTree N) : Prop :=
  t.leaves.Perm (List.finRange N)

/-- Syntax-only translation to the shared transparent helper expansion. -/
def sourceTree {N : Nat} : ReductionTree N → DSL.WelfordSource.Tree N
  | .leaf i => .leaf i
  | .node l r => .node (sourceTree l) (sourceTree r)

end
end VeriTile.Bench.Repairs.FusedLayernormTriton.Kernels

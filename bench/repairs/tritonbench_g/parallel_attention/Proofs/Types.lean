import bench.repairs.tritonbench_g.parallel_attention.Proofs.Graph

namespace VeriTile.Bench.Repairs.ParallelAttention
noncomputable section
open scoped BigOperators

/-- Logical input elements, independent of physical memory, tiles and execution. -/
structure Inputs (α : Type) where
  q : Nat → Nat → α
  k : Nat → Nat → α
  v : Nat → Nat → α

structure Cotangent (α : Type) where
  output : Nat → Nat → α
  normalizer : Nat → α

/-- The independent causal quadratic-attention score. -/
inductive PublicResult (α : Type) where
  | both (output : Nat → Nat → α) (normalizer : Nat → α)
  | single (output : Nat → Nat → α)


def headInputs (x : TensorInputs α) (bh : Nat) : Inputs α := ⟨x.q bh,x.k bh,x.v bh⟩
def headCotangent (x : TensorInputs α) (bh : Nat) : Cotangent α := ⟨x.do_ bh,x.dz bh⟩

end
end VeriTile.Bench.Repairs.ParallelAttention

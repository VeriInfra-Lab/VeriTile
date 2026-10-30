import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicIOLayout

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section

/-- Independent source-ordered scalar and finite-dot formulas. The numerical
operations and dot plans retain their existing primitive/compiler meanings. -/
def forwardFormula {α : Type} (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (xs : ChannelValues α (forwardInputs s)) : ChannelValues α (forwardOutputs s) :=
  Fin.cases (fun j => forwardLane a.ops (plans s p) s
    (forwardLogical a.ops.zero s xs) tile j.1.val j.2.1.val) (fun k => k.elim0)

def backwardFormula {α : Type} (a : Scalar α) (p : DotTrees) (s : Shape) (tile : Tile)
    (xs : ChannelValues α (backwardInputs s)) : ChannelValues α (backwardOutputs s) := by
  let x := backwardLogical a.ops.zero s xs
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro j; exact queryLane a.ops (plans s p) s x tile j.1.val j.2.1.val
  · intro j; exact (keyValueLane a.ops (plans s p) s x tile j.1.val j.2.1.val 0).1
  · intro j; exact (keyValueLane a.ops (plans s p) s x tile j.1.val 0 j.2.1.val).2


end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO

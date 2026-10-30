import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicLogicalModel
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelRealTarget

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO
open VeriTile.Triton FP.Structural Kernel Kernel.Native
noncomputable section

/-- The mathematical target is a causal sum, independent of source execution
and of the ordered native recurrence used in the other public entry. -/
def forwardMath (s : Shape) (tile : Tile) (xs : ChannelValues ℝ (forwardInputs s)) :
    ChannelValues ℝ (forwardOutputs s) :=
  Fin.cases (fun j => RealTarget.forward s (forwardLogical 0 s xs) tile j.1.val j.2.1.val)
    (fun k => k.elim0)

def backwardMath (s : Shape) (tile : Tile) (xs : ChannelValues ℝ (backwardInputs s)) :
    ChannelValues ℝ (backwardOutputs s) := by
  let x := backwardLogical 0 s xs
  refine Fin.cases ?_ (Fin.cases ?_ (Fin.cases ?_ (fun k => k.elim0)))
  · intro j; exact RealTarget.query s x tile j.1.val j.2.1.val
  · intro j; exact RealTarget.key s x tile j.1.val j.2.1.val
  · intro j; exact RealTarget.value s x tile j.1.val j.2.1.val


end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.PublicIO

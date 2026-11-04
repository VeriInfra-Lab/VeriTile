import bench.tritonbench_g.parallel_retention_attention.Proofs.HelperBounds
import bench.tritonbench_g.parallel_attention.Proofs.DescendingRange

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (BackwardConfig descendingStart)

/-- The descending helper also visits blocks above its own output chunk. -/
structure BackwardIndexBounds (c : BackwardConfig) : Prop extends HelperIndexBounds c where
  sourceTime : (c.T+c.BTS-1)/c.BTS*c.BTS+c.BTS≤2147483648

namespace BackwardIndexBounds
theorem descending_lt (h : BackwardIndexBounds c) (n : Nat) :
    descendingStart c n<2147483648 := by
  by_cases zero : c.BTS=0
  · simp [descendingStart,zero]
  · have := h.sourceTime
    have upper : descendingStart c n≤(c.T+c.BTS-1)/c.BTS*c.BTS := by
      unfold descendingStart
      omega
    omega
end BackwardIndexBounds
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel

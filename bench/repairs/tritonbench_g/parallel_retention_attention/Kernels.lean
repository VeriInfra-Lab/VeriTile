import bench.repairs.tritonbench_g.parallel_retention_attention.SourceSyntax
import bench.repairs.tritonbench_g.parallel_retention_attention.QuerySyntax
import bench.repairs.tritonbench_g.parallel_retention_attention.CandidateSyntax
import bench.repairs.tritonbench_g.parallel_retention_attention.ShellSyntax
import bench.repairs.tritonbench_g.parallel_retention_attention.Contract

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section

/-- Pinned original selected Python prefix, before the separately named
assignment-fusion candidates appended in the completion campaign. -/
def sourceSHA256 : String := "678648ade1cedb09c1b0ae40c03155c9ebc09bb812b3cb945a15c98e18ce2ab5"

/-- The shared readable DSL quotations retain source loads, masks, both loop
phases, casts, barriers, and source dot precision. The backward shell expands
its two helper calls at the grid-decoded program indices. -/
abbrev forwardKernel (s : Shape) := Kernel.FPStages.forwardKernel (Kernel.config s 128)
abbrev backwardKernel (s : Shape) (pids : Nat → Nat) :=
  Kernel.FPBackward.source (Kernel.decodedBackwardConfig s pids) "dq"

/-- Actual shared-interpreter execution followed by physical realization of
its masked stores. Values come from the source state. Local dot staging keeps
the live accumulator and finite multiply/add/FMA order explicit. -/
abbrev executeForward := @Kernel.forwardRun
abbrev executeBackward := @Kernel.backwardRun

end
end VeriTile.Bench.Repairs.ParallelRetentionAttention

import bench.bug_detection.tasks.gemma_one_plus_w_bf16.Contract

namespace VeriTile.Bench.BugDetection.gemma_one_plus_w_bf16
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,j) => (1 + xs.weight j) * xs.x r j / Real.sqrt ((∑ k : Fin 2048, (xs.x r k)^2) / 2048 + 1/1000000))

end
end VeriTile.Bench.BugDetection.gemma_one_plus_w_bf16

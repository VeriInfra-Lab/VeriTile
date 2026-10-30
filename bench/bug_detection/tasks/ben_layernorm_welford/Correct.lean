import bench.bug_detection.tasks.ben_layernorm_welford.Contract

namespace VeriTile.Bench.BugDetection.ben_layernorm_welford
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,j) => xs.weight j * (xs.x r j - mean xs r) / Real.sqrt ((∑ k : Fin 2048, (xs.x r k - mean xs r)^2) / 2048 + 1/1000000) + xs.bias j)

end
end VeriTile.Bench.BugDetection.ben_layernorm_welford

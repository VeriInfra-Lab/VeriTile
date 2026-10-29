import bench.bug_detection.tasks.mlp_gelu_exact_vs_tanh.Contract

namespace VeriTile.Bench.BugDetection.mlp_gelu_exact_vs_tanh
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,h) => ∑ j, (activation (gateValue xs r j) * upValue xs r j) * xs.down h j)

end
end VeriTile.Bench.BugDetection.mlp_gelu_exact_vs_tanh

import bench.bug_detection.tasks.ben_gelu_handrolled.Contract

namespace VeriTile.Bench.BugDetection.ben_gelu_handrolled
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,j) => 0.5 * xs.x r j * (1 + Real.tanh (0.7978845608028654 * (xs.x r j + 0.044715 * (xs.x r j * xs.x r j * xs.x r j)))))

end
end VeriTile.Bench.BugDetection.ben_gelu_handrolled

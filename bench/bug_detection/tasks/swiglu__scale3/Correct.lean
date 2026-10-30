import bench.bug_detection.tasks.swiglu__scale3.Contract

namespace VeriTile.Bench.BugDetection.swiglu__scale3
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
end VeriTile.Bench.BugDetection.swiglu__scale3

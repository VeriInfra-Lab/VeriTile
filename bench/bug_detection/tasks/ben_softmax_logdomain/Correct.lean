import bench.bug_detection.tasks.ben_softmax_logdomain.Contract

namespace VeriTile.Bench.BugDetection.ben_softmax_logdomain
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,j) => Real.exp (xs.x r j) / ∑ k : Fin 256, Real.exp (xs.x r k))

end
end VeriTile.Bench.BugDetection.ben_softmax_logdomain

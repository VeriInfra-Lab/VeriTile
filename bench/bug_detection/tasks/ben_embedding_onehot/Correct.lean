import bench.bug_detection.tasks.ben_embedding_onehot.Contract

namespace VeriTile.Bench.BugDetection.ben_embedding_onehot
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,h) => xs.weight (index xs r) h)

end
end VeriTile.Bench.BugDetection.ben_embedding_onehot

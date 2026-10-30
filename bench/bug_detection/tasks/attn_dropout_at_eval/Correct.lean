import bench.bug_detection.tasks.attn_dropout_at_eval.Contract

namespace VeriTile.Bench.BugDetection.attn_dropout_at_eval
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,h,d) => ∑ j : Fin 16, (weight xs h r j / (∑ k : Fin 16, weight xs h r k)) * xs.v (kv h) j d)

end
end VeriTile.Bench.BugDetection.attn_dropout_at_eval

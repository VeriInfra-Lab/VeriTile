import bench.bug_detection.tasks.moe_router_bf16.Contract

namespace VeriTile.Bench.BugDetection.moe_router_bf16
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (t,h) => (∑ j : Fin 6, probability xs t j * expertValue xs t (route xs t j) h) + sharedValue xs t h)

end
end VeriTile.Bench.BugDetection.moe_router_bf16

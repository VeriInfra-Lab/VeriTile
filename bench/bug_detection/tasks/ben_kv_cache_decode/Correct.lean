import bench.bug_detection.tasks.ben_kv_cache_decode.Contract

namespace VeriTile.Bench.BugDetection.ben_kv_cache_decode
open VeriTile Triton
noncomputable section
set_option maxRecDepth 10000
set_option maxHeartbeats 2000000
open Contract
open scoped BigOperators VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved Real specification of the complete original source.
All execution, safety, input/output and frame obligations are retained. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (r,h,d) => ∑ j : Fin 32, (weight xs h r j / (∑ k : Fin 32, weight xs h r k)) * xs.v (kv h) j d)

end
end VeriTile.Bench.BugDetection.ben_kv_cache_decode

import bench.tritonbench_g.context_attn_fwd.NativeFormula
import VeriTile.Triton.Float.ArrayProgramIO

namespace VeriTile.Bench.Repairs.ContextAttnFwd.NativeArrayBridge
open VeriTile.Triton FP.Structural NativeFormula
noncomputable section
variable {α : Type} [Inhabited α]

def invocation (p : Layout) (r : Regions) (scale : ℝ) (M : Algebra α)
    (job : Launch) : Invocation α where
  kernel := kernel p r .fp32 scale
  backend := M.withDefaultPrecision .fp32
  pids := job.pids
  numPids := job.numPids

end
end VeriTile.Bench.Repairs.ContextAttnFwd.NativeArrayBridge

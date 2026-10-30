import bench.tritonbench_g.context_attn_mistral.Proofs.Mathematical_HostReadback
import VeriTile.Triton.Memory.AlgorithmProgramIO
import bench.tritonbench_g.context_attn_mistral.IO

/-! Public regional IO of the complete original ordinary grid. The existing
flat-allocation transport remains a separate physical-memory theorem. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface
open VeriTile Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
open scoped Classical

/-- Independent gathered finite-sentinel weighted sum, with every padded
key contribution, original natural exponential and per-block range retained. -/
def formula (c : Host.Call) (capacity : RegionBounds) (values : Inputs c capacity) :
    Fin 1 → Fin (laneCount c) → ℝ := fun _ j=>
  let k:=(lanes c).symm j
  Formula.expected c.layout (c.launchAt k.1) (logicalInputs c capacity values)
    (Host.scale c.layout.D) k.2.1.val k.2.2.1.val

#specNonCircular formula avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
#specNonCircular Geometry avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
#specNonCircular Metadata avoiding [exec,execR,AlgorithmExecution,Kernel.GridLaunchedOrdinary]
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel.RealInterface

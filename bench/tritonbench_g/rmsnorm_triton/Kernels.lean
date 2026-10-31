import bench.tritonbench_g.rmsnorm_triton.Sources.Selected
import bench.tritonbench_g.rmsnorm_triton.Sources.Mathematical_WholeSource

/-! Current checked source: FP32 input, weight, accumulation and output. Both
1024-lane loops, the integer pow call, explicit casts and masked weight load
are retained. The candidate commutes one integer address product only. -/
namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Triton
abbrev Parameters := VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.Parameters
abbrev Memory := VeriTile.Bench.TritonBenchG.RmsnormTriton.Whole.Memory

def blockSize : Nat := 1024

/- The actual bodies are shared through Sources.Selected. -/
end VeriTile.Bench.Repairs.RmsnormTriton

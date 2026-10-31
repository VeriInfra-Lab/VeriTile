import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_Kernels
import VeriTile.Triton.Float.Erasure
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Algorithm
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 20000

def seedOp : Random.IntegerType → Op .int []
  | .i32 | .i64 => .ref .int [] "seed_arg"
  | .u32 | .u64 => .castNatToInt (.ref .nat [] "seed_arg")

end
end VeriTile.Bench.TritonBenchG.SeededDropout.Algorithm

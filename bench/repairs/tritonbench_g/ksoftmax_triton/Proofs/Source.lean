import bench.repairs.tritonbench_g.ksoftmax_triton.Kernels
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardMath
namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton
def backwardReal (D G O : RegionName) (bm bn gm gn om on K DEPTH : Nat) (log causal : Bool) :=
  (backward D G O bm bn gm gn om on K DEPTH log causal).eraseDType
end VeriTile.Bench.Repairs.Ksoftmax

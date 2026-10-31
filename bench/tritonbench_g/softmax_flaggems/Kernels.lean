import bench.tritonbench_g.softmax_flaggems.Sources.Selected
import bench.tritonbench_g.softmax_flaggems.Sources.Kernels
/-! Shared repaired finite-seed kernels and their existing address candidates. -/

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2
/-- The uncalled Python helper on nonnegative dimensions and a positive divisor. -/
def next_multiple_of (a b : Nat) : Nat := ((a+b-1)/b)*b
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2

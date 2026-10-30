import bench.repairs.tritonbench_g.sin_computation.Kernels
import bench.tritonbench_g.sin_computation.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Projection
open VeriTile.Triton

theorem kernel_algorithm (X O : RegionName) (n B : Nat) :
    (kernel X O n B).toAlgKernel =
      (VeriTile.Bench.TritonBenchG.SinComputation.sin_kernel X O n B).toAlgKernel := by rfl

theorem original_projection (n : Nat) :
    (originalKernel n).toAlgorithm? = .ok (originalKernel n).toAlgKernel := by rfl

#axiomsClean kernel_algorithm
#axiomsClean original_projection
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Projection

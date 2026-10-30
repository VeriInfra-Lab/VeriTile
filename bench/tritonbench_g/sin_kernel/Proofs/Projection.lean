import bench.tritonbench_g.sin_kernel.Kernels
import bench.tritonbench_g.sin_kernel.Proofs.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SinKernel.Projection
open VeriTile.Triton

theorem kernel_algorithm (X O : RegionName) (n B : Nat) :
    (kernel X O n B).toAlgKernel =
      (VeriTile.Bench.TritonBenchG.SinKernel.kernel_function X O n B).toAlgKernel := by rfl

theorem original_projection (n : Nat) :
    (originalKernel n).toAlgorithm? = .ok (originalKernel n).toAlgKernel := by rfl

#axiomsClean kernel_algorithm
#axiomsClean original_projection
end VeriTile.Bench.Repairs.SinKernel.Projection

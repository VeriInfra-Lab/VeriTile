import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Sources.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.VectorAddition.Contiguous.Projection
open VeriTile.Triton
noncomputable section

/-- Real projection only; FP execution uses the source with its precision tags. -/
theorem kernel_algorithm (X Y O : RegionName) (n B : Nat) :
    (add_kernel X Y O n B).toAlgKernel =
      (VeriTile.Bench.TritonBenchG.VectorAddition.add_kernel X Y O n B).toAlgKernel := rfl

theorem original_projection (n : Nat) :
    (originalKernel n).toAlgorithm? = .ok (originalKernel n).toAlgKernel := rfl

#axiomsClean kernel_algorithm
#axiomsClean original_projection
end
end VeriTile.Bench.Repairs.VectorAddition.Contiguous.Projection

import bench.tritonbench_g.add_value.Kernels
import bench.tritonbench_g.add_value.Proofs.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.Projection
open VeriTile.Triton
noncomputable section

/-- Mathematical projection of the shared explicit-FP32 source. This is used
only by the Real proofs, never as justification for FP transformation semantics. -/
theorem kernel_projection (X O : RegionName) (n B : Nat) :
    (kernel X O n B).toAlgorithm? =
      Except.ok (VeriTile.Bench.TritonBenchG.AddValue.puzzle1_kernel X O n B).toAlgKernel := by
  rfl

theorem kernel_algorithm (X O : RegionName) (n B : Nat) :
    (kernel X O n B).toAlgKernel =
      (VeriTile.Bench.TritonBenchG.AddValue.puzzle1_kernel X O n B).toAlgKernel := by
  rfl

theorem original_projection (n : Nat) :
    (originalKernel n).toAlgorithm? = Except.ok (originalKernel n).toAlgKernel := by
  rfl

#axiomsClean kernel_projection
#axiomsClean kernel_algorithm
#axiomsClean original_projection
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.Projection

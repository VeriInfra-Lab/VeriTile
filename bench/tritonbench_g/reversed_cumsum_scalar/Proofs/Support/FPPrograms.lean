import bench.tritonbench_g.reversed_cumsum_scalar.Kernels
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.Rewrite
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.FPExecution

/-! The intended input-index commutation on the complete repaired kernel.
All floating operations and their precision remain in their original context. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (h : Call) (c : Config) : Program := ⟨Kernels.sourceKernel h c,[]⟩
def indexCommutedKernel (h : Call) (c : Config) : Program := ⟨Kernels.indexCommutedKernel h c,[]⟩

end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Equivalence

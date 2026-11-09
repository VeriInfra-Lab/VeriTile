import bench.tritonbench_g.reversed_cumsum.Kernels
import VeriTile.Triton.Float.ProfiledRewrite

/-! The two actual typed sources and their empty numerical assumption list. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.Equivalence
open VeriTile Triton FP.ProfiledRewrite
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (h : Call) (c : Config) : Program := ⟨Kernels.sourceKernel h c,[]⟩
def compareRightKernel (h : Call) (c : Config) : Program := ⟨Kernels.compareRightKernel h c,[]⟩

end
end VeriTile.Bench.Repairs.ReversedCumsum.Equivalence

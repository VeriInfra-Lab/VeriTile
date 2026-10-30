import bench.repairs.tritonbench_g.rmsnorm_implementation.Kernels
import VeriTile.Triton.Float.ProfiledMemory

namespace VeriTile.Bench.Repairs.RmsnormImplementation.FPInterface
open VeriTile Triton FP.ProfiledMemory
open scoped VeriTile.Spec
noncomputable section

def assumptions : Spec.Assumptions FP.GuardedFragment := []

def original (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) : Program where
  kernel := kernel "x" "weights" "out" sxb sxm sxk sw sob som sok n B eps
  domain := []
def commuted (sxb sxm sxk sw sob som sok n B : Nat) (eps : ℝ) : Program where
  kernel := indexCommutedKernel "x" "weights" "out" sxb sxm sxk sw sob som sok n B eps
  domain := []

end
end VeriTile.Bench.Repairs.RmsnormImplementation.FPInterface

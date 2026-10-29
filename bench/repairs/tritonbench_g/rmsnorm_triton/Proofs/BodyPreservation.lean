import bench.repairs.tritonbench_g.rmsnorm_triton.Kernels
import bench.repairs.tritonbench_g.rmsnorm_triton.Proofs.LegacyBody

namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile.Triton

theorem kernel_preserved (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    kernel X W O sxb sxm sxk sw sob som sok N B eps = LegacyBody.kernel X W O sxb sxm sxk sw sob som sok N B eps := rfl

theorem indexCommutedKernel_preserved (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    indexCommutedKernel X W O sxb sxm sxk sw sob som sok N B eps = LegacyBody.indexCommutedKernel X W O sxb sxm sxk sw sob som sok N B eps := rfl

end VeriTile.Bench.Repairs.RmsnormTriton

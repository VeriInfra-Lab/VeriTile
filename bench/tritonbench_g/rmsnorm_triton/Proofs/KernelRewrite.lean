import bench.tritonbench_g.rmsnorm_triton.IO
import bench.tritonbench_g.cache_transform_triton.Sources.Mathematical_IndexRewrite

namespace VeriTile.Bench.Repairs.RmsnormTriton
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000

def originalProgram (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : Program :=
  ⟨kernel X W O sxb sxm sxk sw sob som sok N B eps,[]⟩
def candidateProgram (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) : Program :=
  ⟨indexCommutedKernel X W O sxb sxm sxk sw sob som sok N B eps,[]⟩

namespace KernelRewrite
/-- Both complete source loops retain every floating backend invocation.
Only natural-number row-address products commute. -/
theorem equivalent (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) :
    originalProgram X W O sxb sxm sxk sw sob som sok N B eps ≡[([] : Spec.Assumptions FP.GuardedFragment)]
      candidateProgram X W O sxb sxm sxk sw sob som sok N B eps := by
  equiv_decompose <;> fp_prove

#axiomsClean equivalent
end KernelRewrite
end
end VeriTile.Bench.Repairs.RmsnormTriton

import bench.repairs.tritonbench_g.triton_mul2.FPEquiv

/-! Optional source-binding checks for the public FP programs.
The original declaration names are retained outside the reader entry. -/
namespace VeriTile.Bench.Repairs.TritonMul2.FPEquiv
open VeriTile.Triton FP.ProfiledMemory
noncomputable section

theorem original_source (X O : RegionName) (n B : Nat) :
    (original X O n B).kernel = sourceKernel X O n B := rfl

theorem inplace_source (X : RegionName) (n B : Nat) :
    (inplaceOriginal X n B).kernel = inplaceSourceKernel X n B := rfl

#axiomsClean original_source
#axiomsClean inplace_source
end
end VeriTile.Bench.Repairs.TritonMul2.FPEquiv

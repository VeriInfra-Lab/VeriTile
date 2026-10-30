import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.PublicReal

namespace VeriTile.Bench.Repairs.F8ConversionUtils.Correct
open VeriTile.Triton
open scoped MaskedKernelIO₁
noncomputable section

/-- The selected source profile determines both storage widths and casts.
Real interpretation erases those floating tags; every active lane keeps the
same value. Decode retains both source stores. No FP8 round-trip claim. -/
def conversionIO (p : Source.Profile) (Y X : RegionName) (N B : Nat) : MaskedKernelIO₁ :=
  ConversionIO.real p Y X N B

specification complete_kernel_real_formula (p : Source.Profile)
    (Y X : RegionName) (N B : Nat) :
    VeriTile.Spec.Real (conversionIO p Y X N B ⊨ fun xs j => xs j) :=
  PublicReal.implements p Y X N B

#axiomsClean complete_kernel_real_formula
#stmtSurfaceSubset complete_kernel_real_formula ⊆
  [VeriTile.Spec.Real, Source.Profile, RegionName, conversionIO,
   MaskedKernelIO₁.Implements, MaskedKernelIO₁.B, Checked.conversionReal]

end
end VeriTile.Bench.Repairs.F8ConversionUtils.Correct

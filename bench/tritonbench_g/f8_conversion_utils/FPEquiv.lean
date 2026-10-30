import bench.tritonbench_g.f8_conversion_utils.Proofs.SourceEquivalence

namespace VeriTile.Bench.Repairs.F8ConversionUtils.FPEquiv
open VeriTile Triton
open scoped VeriTile.Spec

abbrev assumptions := SourceEquivalence.assumptions
abbrev original := SourceEquivalence.original
abbrev indexCommuted := SourceEquivalence.indexCommuted

/-- The two complete typed conversion sources differ only in integer index
multiplication operands. All six storage profiles, masks, both decoder stores and
source casts are retained, with no floating transformation assumption. -/
specification conversion_index_equiv (p : Source.Profile) (Y X : RegionName) (N B : Nat) :
    original p Y X N B ≡[assumptions] indexCommuted p Y X N B :=
  SourceEquivalence.equivalent p Y X N B

#print_fp_assumptions conversion_index_equiv
#axiomsClean conversion_index_equiv
end VeriTile.Bench.Repairs.F8ConversionUtils.FPEquiv

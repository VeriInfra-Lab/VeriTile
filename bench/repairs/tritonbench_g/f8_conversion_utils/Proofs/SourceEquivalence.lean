import bench.repairs.tritonbench_g.f8_conversion_utils.Kernels
import bench.tritonbench_g.f8_conversion_utils.Proofs.FP

namespace VeriTile.Bench.Repairs.F8ConversionUtils.SourceEquivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec

def assumptions : Spec.Assumptions FP.GuardedFragment := []
def original (p : Source.Profile) (Y X : RegionName) (N B : Nat) : Program :=
  ⟨Source.kernel p Y X N B, []⟩
def indexCommuted (p : Source.Profile) (Y X : RegionName) (N B : Nat) : Program :=
  ⟨Source.indexCommuted p Y X N B, []⟩

/-- Reuse the exact integer-index proof for each existing source profile. -/
theorem equivalent (p : Source.Profile) (Y X : RegionName) (N B : Nat) :
    original p Y X N B ≡[assumptions] indexCommuted p Y X N B := by
  cases p
  · exact VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof.to_half_equivalent .e5 Y X N B
  · exact VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof.to_fp8_equivalent .fp16 .e5 Y X N B
  · exact VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof.to_fp8_equivalent .fp32 .e5 Y X N B
  · exact VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof.to_half_equivalent .e4 Y X N B
  · exact VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof.to_fp8_equivalent .fp16 .e4 Y X N B
  · exact VeriTile.Bench.TritonBenchG.F8ConversionUtils.FPProof.to_fp8_equivalent .fp32 .e4 Y X N B

#print_fp_assumptions equivalent
#axiomsClean equivalent
end VeriTile.Bench.Repairs.F8ConversionUtils.SourceEquivalence

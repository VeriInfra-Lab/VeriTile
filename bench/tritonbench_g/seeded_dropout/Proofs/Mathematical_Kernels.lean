import bench.tritonbench_g.seeded_dropout.Kernels
import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_RandomDomain
/-! Compatibility invocation for the shared source; LegalBlock supplies its
existing source arange representability proof. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource
open VeriTile.Triton
abbrev sourceType (seedKind : Random.IntegerType) :=
  RegionName → RegionName → Nat → ℝ → SeedCarrier seedKind →
    (B : Nat) → SourceDomain.LegalBlock B → ComputeKernel

def source (storage : Storage) (seedKind : Random.IntegerType) (swapped : Bool) :
    sourceType seedKind :=
  fun x y n p seed B hB => VeriTile.Bench.Repairs.SeededDropout.sourceFamily
    storage seedKind swapped x y n p seed B (RandomDomain.arange_domain hB)
end VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource

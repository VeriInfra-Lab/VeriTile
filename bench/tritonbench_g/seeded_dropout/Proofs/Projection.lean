import bench.tritonbench_g.seeded_dropout.Kernels
import bench.tritonbench_g.seeded_dropout.Contract
import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_WholeMath

namespace VeriTile.Bench.Repairs.SeededDropout
open VeriTile.Triton

def legacyKind : SeedKind → Random.IntegerType
  | .i32 => .i32 | .i64 => .i64 | .u64 => .u64

def legacySeed : (k : SeedKind) → SeedCarrier k → TritonBenchG.SeededDropout.TypedSource.SeedCarrier (legacyKind k)
  | .i32,s | .i64,s | .u64,s => s

theorem source_identity (k : SeedKind) (x y : RegionName) (n : Nat) (p : ℝ) (s : SeedCarrier k) :
    kernel k x y n p s = TritonBenchG.SeededDropout.TypedSource.source .fp32 (legacyKind k) Bool.false x y n p
      (legacySeed k s) 1024 TritonBenchG.SeededDropout.SourceDomain.wrapper_block_legal := by cases k <;> rfl

theorem legacy_seed_value (k : SeedKind) (s : SeedCarrier k) :
    TritonBenchG.SeededDropout.TypedSource.seedInt (legacyKind k) (legacySeed k s) = seedValue k s := by cases k <;> rfl

theorem legacy_seed_abi (k : SeedKind) (s : SeedCarrier k) (h : SeedABI k s) :
    TritonBenchG.SeededDropout.Whole.PythonSeedABI (legacyKind k) (legacySeed k s) := by
  cases k <;> simpa [SeedABI,TritonBenchG.SeededDropout.Whole.PythonSeedABI,legacyKind,legacySeed,
    seedValue,TritonBenchG.SeededDropout.TypedSource.seedInt] using h
#axiomsClean source_identity
end VeriTile.Bench.Repairs.SeededDropout

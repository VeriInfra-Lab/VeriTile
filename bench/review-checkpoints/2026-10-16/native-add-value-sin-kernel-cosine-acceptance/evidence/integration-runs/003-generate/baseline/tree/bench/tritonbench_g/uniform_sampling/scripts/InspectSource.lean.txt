import bench.tritonbench_g.uniform_sampling.Kernels

/-! Read-only inspection of actual expanded source statements. Parameters
remain symbolic, so this does not precompute any random output. -/
namespace VeriTile.Bench.TritonBenchG.UniformSampling.SourceInspection
open VeriTile.Triton TypedSource

#reduce fun (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : ℝ)
  (B : Nat) (hB : SourceDomain.LegalBlock B) =>
  (source .fp64 Bool.false y N seed offset lower upper B hB).surfaceBody[8]?

#reduce fun (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : ℝ)
  (B : Nat) (hB : SourceDomain.LegalBlock B) =>
  (source .fp64 Bool.false y N seed offset lower upper B hB).surfaceBody[11]?

#reduce fun (y : RegionName) (N : Nat) (seed offset : Int) (lower upper : ℝ)
  (B : Nat) (hB : SourceDomain.LegalBlock B) =>
  (source .fp64 Bool.false y N seed offset lower upper B hB).surfaceBody[19]?

end VeriTile.Bench.TritonBenchG.UniformSampling.SourceInspection

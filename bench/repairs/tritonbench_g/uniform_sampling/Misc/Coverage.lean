import bench.tritonbench_g.uniform_sampling.Proofs.SourceDomain

/-! Optional launch coverage application, separate from the per-program
mathematical and floating implementation-pair specifications. -/
namespace VeriTile.Bench.Repairs.UniformSampling.Correct
open VeriTile.Bench.TritonBenchG.UniformSampling

/-- Every requested coordinate has a unique active program, field and lane. -/
theorem output_coverage {N B k : Nat} (hB : 0 < B) (hk : k < N) :
    ∃! pfl : Fin (SourceDomain.grid N B) × Fin 4 × Fin B,
      SourceDomain.outputIndex pfl.1.val B pfl.2.1 pfl.2.2 = k :=
  SourceDomain.active_coverage_unique hB hk

#axiomsClean output_coverage
end VeriTile.Bench.Repairs.UniformSampling.Correct

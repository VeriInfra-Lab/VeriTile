import bench.tritonbench_g.context_attn_bloom.Proofs.ProgramPair

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
open scoped VeriTile.Spec
noncomputable section

/-- Complete actual metadata-selected grids, comparing the original with row
origin reuse. The original WordISA, all-word payloads, metadata domain, masks,
FP32 casts, TF32/FMA graph, finite sentinel and guard remain unchanged. Both
runs are safe and successful, every native observed output agrees, and both
exact raw exterior frames hold, including the original empty-grid branch. -/
specification source_candidate_equivalence (g : Raw.Geometry) (r : Regions)
    (extent : RegionBounds) (ops : Native.WordISA) :
    PairIO.original g r extent ops ≡[[]] PairIO.indexShared g r extent ops := by
  exact Spec.FloatingPoint.ofStructural rfl rfl (PairIO.equivalent g r extent ops)

#axiomsClean source_candidate_equivalence
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicKernel

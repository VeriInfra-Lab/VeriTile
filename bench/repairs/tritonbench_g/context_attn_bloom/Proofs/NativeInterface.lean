import bench.repairs.tritonbench_g.context_attn_bloom.NativeIO
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
noncomputable section

/-- Complete source safety, full grid execution, independent native observation
and exact exterior cells. The pure observer preserves the source and domain. -/
theorem observes (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (ops : Native.WordISA) :
    (io g r extent).ObservesOn (observe g r extent ops)
      (KernelIO.domain g r extent) (formula g r extent ops) :=
  ProgramIO.ObservesOn.of_implementsOn
    (KernelIO.implements g r extent Native.scalar Native.dots) (observe g r extent ops)

#axiomsClean observes
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.NativeIO

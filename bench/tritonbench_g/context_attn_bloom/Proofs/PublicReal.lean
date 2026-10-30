import bench.tritonbench_g.context_attn_bloom.Domains
import bench.tritonbench_g.context_attn_bloom.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
noncomputable section

/-- Explicit empty-grid case preserves original calls whose unused metadata
lies outside all declared input capacities. Nonempty calls use the unchanged
original logical conditions and the already proved complete grid theorem. -/
theorem public_implements (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) :
    (io g r extent).ImplementsOn (PublicDomain.real g r extent) (formula g r extent) := by
  intro xs domain s hin
  rcases domain with empty|valid
  · letI : IsEmpty (GridIndex (Host.grid g.arguments)):=empty
    have jobs:=PublicDomain.jobs_empty g empty
    have stages : (io g r extent).stages xs=[] := by simp only [io,jobs,List.map_nil]
    rw [stages]
    refine ⟨AlgorithmInitialUndef.zero.initial s,.nil _,fun _ _=>trivial,?_,?_⟩
    · intro o i _
      exact isEmptyElim (KernelIO.coordinate g i).1
    · intro _ _ _ _
      rfl
  · exact implements g r extent xs valid s hin

#axiomsClean public_implements
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.RealKernelIO

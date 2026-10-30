import bench.repairs.tritonbench_g.context_attn_bloom.Domains
import bench.repairs.tritonbench_g.context_attn_bloom.Proofs.NativeInterface

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.NativeIO
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
noncomputable section

/-- Empty grids have no source reads and no output coordinates. This avoids
restricting previously admitted unused signed metadata outside finite windows. -/
theorem public_observes (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (ops : Native.WordISA) :
    (io g r extent).ObservesOn (observe g r extent ops)
      (PublicDomain.native g r extent) (formula g r extent ops) := by
  intro xs domain bounds hb s hin
  rcases domain with empty|valid
  · letI : IsEmpty (GridIndex (Host.grid g.arguments)):=empty
    have jobs:=PublicDomain.jobs_empty g empty
    have schedule : (io g r extent).schedule xs=[] := by
      simp only [io,KernelIO.io,KernelIO.order,jobs]
      split <;> rfl
    rw [schedule]
    refine ⟨trivial,s,rfl,?_,fun _ _ _=>rfl⟩
    intro o i _
    fin_cases o
    exact isEmptyElim (KernelIO.coordinate g i.1).1
  · exact observes g r extent ops xs valid bounds hb s hin

#axiomsClean public_observes
end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.NativeIO

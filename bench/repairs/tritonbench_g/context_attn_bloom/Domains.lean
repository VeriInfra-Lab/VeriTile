import bench.repairs.tritonbench_g.context_attn_bloom.IO
import bench.repairs.tritonbench_g.context_attn_bloom.NativeIO

namespace VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicDomain
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnBloom.Program
noncomputable section

def native (g : Raw.Geometry) (r : Regions) (extent : RegionBounds) (xs : (NativeIO.io g r extent).Inputs) : Prop :=
  empty g ∨ KernelIO.domain g r extent xs

theorem jobs_empty (g : Raw.Geometry) (h : empty g) : KernelIO.jobs g=[] := by
  letI : IsEmpty (GridIndex (Host.grid g.arguments)):=h
  cases equal : KernelIO.jobs g with
  | nil=>rfl
  | cons idx rest=>exact isEmptyElim idx

end
end VeriTile.Bench.Repairs.ContextAttnBloomWhole001.PublicDomain

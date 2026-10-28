import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Kernels
import bench.repairs.tritonbench_g.layer_norm_fwd.Proofs.SourceBinding

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd

/-- Actual canonical source with the existing public parameter records. -/
def actual (f : Flags) (r : Regions) (stride : Strides) (M N : Nat)
    (eps p : ℝ) (block : Nat) (hb : Random.ArangeSizeDomain block) : ComputeKernel :=
  Source.kernel (sourceFlags f) (sourceRegions r) (sourceStrides stride) M N eps p block hb
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.TypedSource

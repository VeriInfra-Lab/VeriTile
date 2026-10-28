import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.Pair
import bench.repairs.tritonbench_g.layer_norm_fwd_nonzero_tile.Proofs.RawPairSafety

namespace VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Pair
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.TritonBenchG.LayerNormFwd
open VeriTile.Bench.Repairs.LayerNormFwd
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 6000

/-- The changed two-statement prefix only forms an integer pointer; neither
order of the row/stride multiplication accesses memory. -/
theorem candidate_prefix_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList A bounds
      ((indexCommuted f r stride M N eps p).kernel.surfaceBody.take 2) s :=
  RawPair.candidate_prefix_safe A (sourceFlags f) (sourceRegions r) (sourceStrides stride)
    M N eps p (tile N) (tile_arange N) bounds s

/-- Safety is transported through the pure prefix's equal actual successor,
then through the literally identical complete suffix. Execution equality alone
is not used as a safety certificate. -/
theorem trace_safe {α : Type} [Inhabited α] (A : Algebra α)
    (f : Flags) (r : Regions) (stride : Strides) (M N : Nat) (eps p : ℝ)
    (bounds : RegionName → Nat) (s : State α)
    (safe : TraceSafe A bounds (original f r stride M N eps p).kernel s) :
    TraceSafe A bounds (indexCommuted f r stride M N eps p).kernel s :=
  RawPair.trace_safe A (sourceFlags f) (sourceRegions r) (sourceStrides stride)
    M N eps p (tile N) (tile_arange N) bounds s safe

#axiomsClean trace_safe
end
end VeriTile.Bench.Repairs.LayerNormFwdNonzeroTile.Pair

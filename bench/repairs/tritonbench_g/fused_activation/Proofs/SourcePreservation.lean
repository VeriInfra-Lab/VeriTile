import bench.repairs.tritonbench_g.fused_activation.Kernels
import bench.repairs.tritonbench_g.fused_activation.Proofs.LegacyPastedSurface
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.FusedActivation.SourcePreservation
open VeriTile.Triton
set_option maxRecDepth 65536
set_option maxHeartbeats 4000000
/-- Complete kernel identity on both valid source string alternatives, under
all pointer/size/multiplier values; no floating identity or new premise. -/
theorem original (X B I : RegionName) (W N BS : Nat) (m : ℝ) (a : Bool) :
    kernel X B I W N BS m a = LegacySource.kernel X B I W N BS m a := by
  cases a <;> rfl
theorem candidate (X B I : RegionName) (W N BS : Nat) (m : ℝ) (a : Bool) :
    commutedKernel X B I W N BS m a = LegacySource.commutedKernel X B I W N BS m a := by
  cases a <;> rfl
#axiomsClean original
#axiomsClean candidate
end VeriTile.Bench.Repairs.FusedActivation.SourcePreservation

import bench.repairs.tritonbench_g.rotary_transform_ops.Proofs.RealWrapper

namespace VeriTile.Bench.Repairs.RotaryTransformOps.HistoricalPublic
open Checked
noncomputable section

theorem rotary_transform_ops_correct (c : Call) (out : Binding.Output c)
    (initial : Physical.Bytes) (accepted : check c initial=Bool.true)
    (callerDevice : Nat) (interpret : Native.Word → ℝ) :
    RealClaim c.host callerDevice initial (realView interpret initial) (Host.mathematical c.host callerDevice initial (realView interpret initial)) := by
  apply whole_real c.host initial
  apply admitted c initial accepted
  intro hi
  have h := out.freshIdentity hi
  rw [←h.1]
  exact Ne.symm h.2

/-- Complete guarded arbitrary-Real mathematical source interpretation. -/
theorem rotary_transform_ops_correct_real (c : Call) (out : Binding.Output c)
    (metadata : Physical.Bytes) (accepted : check c metadata=Bool.true)
    (callerDevice : Nat) (m : Memory ℝ) :
    RealClaim c.host callerDevice metadata m (Host.mathematical c.host callerDevice metadata m) := by
  apply whole_arbitrary_real c.host metadata
  apply admitted c metadata accepted
  intro hi
  have h := out.freshIdentity hi
  rw [←h.1]
  exact Ne.symm h.2

#axiomsClean rotary_transform_ops_correct_real
#axiomsClean rotary_transform_ops_correct
end
end VeriTile.Bench.Repairs.RotaryTransformOps.HistoricalPublic

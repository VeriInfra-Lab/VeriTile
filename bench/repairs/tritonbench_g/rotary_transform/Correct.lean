import bench.repairs.tritonbench_g.rotary_transform.Proofs.RealWrapper

namespace VeriTile.Bench.Repairs.RotaryTransform
open Checked
noncomputable section

theorem rotary_transform_correct (c : Call) (out : Binding.Output c)
    (initial : Physical.Bytes) (accepted : check c initial=Bool.true)
    (callerDevice : Nat) (interpret : Native.Word → ℝ) :
    Public.RealClaim c.host c.views callerDevice interpret initial (Public.ideal c.host c.views callerDevice interpret initial) := by
  apply Public.whole_real c.host c.views initial
  apply admitted c initial accepted
  intro hi
  have h := out.freshIdentity hi
  rw [←h.1]
  exact h.2

/-- Complete mathematical wrapper for independent arbitrary real payloads
and coefficients, with integer metadata checked from caller bytes. -/
theorem rotary_transform_correct_real (c : Call) (out : Binding.Output c)
    (metadata : Physical.Bytes) (accepted : check c metadata=Bool.true)
    (callerDevice : Nat) (m : Memory ℝ) :
    RealWrapper.Claim c out callerDevice metadata m (RealWrapper.invoke c callerDevice metadata m) :=
  RealWrapper.complete c out callerDevice metadata accepted m

#axiomsClean rotary_transform_correct_real
#axiomsClean rotary_transform_correct
end
end VeriTile.Bench.Repairs.RotaryTransform

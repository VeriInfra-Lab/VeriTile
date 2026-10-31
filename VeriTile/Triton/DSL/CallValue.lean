import Mathlib.Data.Real.Basic

namespace VeriTile.Triton.DSL.CallValue
/-- Elaboration provenance for an immutable runtime FP32 argument. The call
frontend checks every resulting use has FP32 evaluation before erasing it. -/
noncomputable def fp32 (value : ℝ) : ℝ := value

/-- Invocation integers retain integer-to-float conversion when used in
floating arithmetic. They are not weak Python floating literals. -/
@[simp] def natural (value : Nat) : Nat := value
@[simp] def integer (value : Int) : Int := value
end VeriTile.Triton.DSL.CallValue

import Mathlib.Data.Real.Basic

namespace VeriTile.Triton.DSL.CallValue
/-- Elaboration provenance for an immutable runtime FP32 argument. The call
frontend checks every resulting use has FP32 evaluation before erasing it. -/
noncomputable def fp32 (value : ℝ) : ℝ := value
end VeriTile.Triton.DSL.CallValue

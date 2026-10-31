import VeriTile.Triton

/-! The complete unmasked scaling source and a scalar-left candidate. The FP
entry selects fp32 input and arithmetic, with the wrapper's fp32 output.
The Python wrapper itself does not constrain the input dtype. Integer addresses are Nat;
launch scheduling and fixed-width address overflow remain outside this model. -/

namespace VeriTile.Bench.TritonBenchG.MulExponentCompensator
open VeriTile.Triton

/-- The constexpr multiplier from `mul_exponent_compensator.py`. -/
noncomputable def exponentCompensator : ℝ :=
  (2 : ℝ) ^ (127 - 15)

end VeriTile.Bench.TritonBenchG.MulExponentCompensator

import VeriTile.Triton.DSL.CallSignature

namespace VeriTile.Tests.PythonLiteralArithmetic
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("x",.fp32),("y",.fp32)]
  scalars := []
  constexpr := []

def folded (x y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(x)
    result = 3 * 0.044715 * value
    tl.store(y, result)
  }

def literal (x y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(x)
    result = 0.134145 * value
    tl.store(y, result)
  }

/-- Python folds the scalar-only product before promoting it into tensor
arithmetic. No runtime FP multiplication is used for `3 * 0.044715`. -/
example (x y : RegionName) : folded x y = literal x y := by rfl

/- Unsupported host cancellation must not become runtime tensor arithmetic. -/
/-- error: zero or cancellation in Python floating literal arithmetic is outside the checked subset -/
#guard_msgs in
example (x y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    value = tl.load(x)
    result = (1.0 - 1.0) * value
    tl.store(y, result)
  }

end VeriTile.Tests.PythonLiteralArithmetic

import VeriTile.Triton.DSL.CallSignature
open VeriTile.Triton
def numericalCaller (X : RegionName) : ComputeKernel :=
  let B := 64
  with_triton_signature ({pointers := [("X",.fp32)], constexpr := ["B"]} : CallSignature) in
  triton {
    lanes = tl.arange(0, B)
    value = tl.load(X + lanes)
    tl.store(X + lanes, value)
  }

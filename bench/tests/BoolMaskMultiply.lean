import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.BoolMaskMultiply
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Mask",.boolean),("Y",.fp32)]

def source (X Mask Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in
  triton {
    x = tl.load(X)
    enabled = tl.load(Mask)
    scaled = x * enabled
    tl.store(Y, scaled)
  }

/-- The floating multiplication is retained even when the mask is false.
This matters for nonfinite x; selecting x/0 would be a different operation. -/
theorem multiplication_retained (X Mask Y : RegionName) :
    source X Mask Y = (triton {
      tl.region X = tl.float32, Mask = tl.int1, Y = tl.float32
      x = tl.load(X)
      enabled = tl.load(Mask)
      scaled = x * tl.where(enabled, $((1 : ℝ)), $((0 : ℝ)))
      tl.store(Y, scaled)
    } : ComputeKernel) := rfl

def vectorSource (X Mask Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature ({signature with constexpr := ["B"]} : CallSignature) in
  triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    enabled = tl.load(Mask + offsets)
    scaled = x * enabled
    tl.store(Y + offsets, scaled)
  }

theorem vector_multiplication_retained (X Mask Y : RegionName) (B : Nat) :
    vectorSource X Mask Y B = (triton {
      tl.region X = tl.float32, Mask = tl.int1, Y = tl.float32
      offsets = tl.arange(0, B)
      x = tl.load(X + offsets)
      enabled = tl.load(Mask + offsets)
      scaled = x * tl.where(enabled, $((1 : ℝ)), $((0 : ℝ)))
      tl.store(Y + offsets, scaled)
    } : ComputeKernel) := rfl

#axiomsClean multiplication_retained
#axiomsClean vector_multiplication_retained
end VeriTile.Tests.BoolMaskMultiply

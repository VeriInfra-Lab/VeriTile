import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FP32CallValue
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("gain",.fp32)]

def source (X Y : RegionName) (gain : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [gain := gain] in
  triton {
    x = tl.load(X)
    y = x * gain
    tl.store(Y, y)
  }

def expanded (X Y : RegionName) (gain : ℝ) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  x = tl.load(X)
  y = x * $((gain : ℝ))
  tl.store(Y, y)
}

theorem fp32_context (X Y : RegionName) (gain : ℝ) :
    source X Y gain = expanded X Y gain := rfl

/-- error: per-call FP32 literal requires an FP32 computation context; use runtime scalar binding -/
#guard_msgs in
def float64Context (X Y : RegionName) (gain : ℝ) : ComputeKernel :=
  with_triton_call ({pointers := [("X",.fp64),("Y",.fp64)], scalars := [("gain",.fp32)]} : CallSignature)
    call_args [gain := gain] in
  triton {
    x = tl.load(X)
    y = x * gain
    tl.store(Y, y)
  }

/-- error: per-call FP32 literal requires an FP32 computation context; use runtime scalar binding -/
#guard_msgs in
def narrowContext (Y : RegionName) (gain : ℝ) : ComputeKernel :=
  with_triton_call ({pointers := [("Y",.fp16)], scalars := [("gain",.fp32)]} : CallSignature)
    call_args [gain := gain] in
  triton {
    tl.store(Y, gain)
  }

/-- error: per-call FP32 literal must have a Real carrier -/
#guard_msgs in
def integerCarrier (X Y : RegionName) (gain : Nat) : ComputeKernel :=
  with_triton_call signature call_args [gain := gain] in
  triton {
    x = tl.load(X)
    tl.store(Y, x * gain)
  }

#axiomsClean fp32_context
end VeriTile.Tests.FP32CallValue

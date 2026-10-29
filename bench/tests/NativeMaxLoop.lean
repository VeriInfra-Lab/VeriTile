import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.NativeMaxLoop
open VeriTile.Triton

def profile : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["N"]

def source (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature profile in
  triton {
    peak = tl.full((), 0, tl.float32)
    acc = tl.full((), 0, tl.float32)
    for i in range(N):
        x = tl.load(X + i)
        peak = tl.maximum(peak, x)
        acc = acc + libdevice.exp(tl.minimum(peak, x))
    tl.store(Y, acc)
  }

def expanded (X Y : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  peak = tl.full([], 0, tl.float32)
  acc = tl.full([], 0, tl.float32)
  tl.for i in $(N) {
    x = tl.load(X + i)
    peak = tl.maximum(peak, x)
    acc = acc + libdevice.exp(tl.minimum(peak, x))
  }
  tl.store(Y, acc)
}

/-- Stable FP32 loop state flows through native min/max and libdevice.exp.
The store retains its original arithmetic value without an invented cast. -/
theorem complete_source (X Y : RegionName) (N : Nat) :
    source X Y N = expanded X Y N := rfl

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def mixedLoop (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature profile in
  triton {
    peak = tl.full((), 0, tl.float32)
    for i in range(N):
        x = tl.load(X + i)
        peak = tl.maximum(peak, x.to(tl.float64))
    tl.store(Y, peak)
  }

#axiomsClean complete_source
end VeriTile.Tests.NativeMaxLoop

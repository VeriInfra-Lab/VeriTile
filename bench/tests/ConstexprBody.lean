import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ConstexprBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B","eps"]

def source (X Y : RegionName) (B : Nat) (eps : ℝ) : ComputeKernel :=
  with_triton_signature signature in
  triton {
    z = tl.zeros((B,), tl.float32)
    spare = tl.full((B, 2), 0.0, tl.float32)
    scalar = tl.full((), 0.0, tl.float32)
    x = tl.load(X + tl.arange(0, B))
    v = tl.sum(x, axis=0) / B
    out = x / tl.sqrt(v + eps)
    tl.store(Y + tl.arange(0, B), out)
  }

def expanded (X Y : RegionName) (B : Nat) (eps : ℝ) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  z = tl.zeros([B], tl.float32)
  spare = tl.full([B, 2], 0.0, tl.float32)
  scalar = tl.full([], 0.0, tl.float32)
  x = tl.load(X + tl.arange(0, B))
  v = tl.sum(x, axis=0) / $((B : Nat))
  out = x / tl.sqrt(v + $((eps : ℝ)))
  tl.store(Y + tl.arange(0, B), out)
}

/-- Only tuple shape spelling and external constexpr values are specialized.
The weak float scalar is consumed at the source operation's FP32 precision. -/
theorem tuple_and_constexpr (X Y : RegionName) (B : Nat) (eps : ℝ) :
    source X Y B eps = expanded X Y B eps := rfl

/-- The same weak scalar must retain FP64 when used by an FP64 tensor. -/
def fp64Source (X Y : RegionName) (eps : ℝ) : ComputeKernel :=
  with_triton_signature
    ({pointers := [("X",.fp64),("Y",.fp64)], constexpr := ["eps"]} : CallSignature) in
  triton {
    x = tl.load(X)
    y = x + eps
    tl.store(Y, y.to(tl.float64))
  }

theorem fp64_constexpr (X Y : RegionName) (eps : ℝ) :
    fp64Source X Y eps = (triton {
      tl.region X = tl.float64, Y = tl.float64
      x = tl.load(X)
      y = x + $((eps : ℝ))
      tl.store(Y, y.to(tl.float64))
    } : ComputeKernel) := rfl

/-- error: constructor shape requires a tuple or list -/
#guard_msgs in
def groupedScalarIsNotTuple (B : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["B"]} : CallSignature) in
  triton {
    x = tl.zeros((B), tl.float32)
  }

/-- error: loop iterator shadows a constexpr parameter -/
#guard_msgs in
def shadowedConstexpr (B : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["B"]} : CallSignature) in
  triton {
    for B in range(3):
        x = B
    y = B
  }

#axiomsClean tuple_and_constexpr
#axiomsClean fp64_constexpr
end VeriTile.Tests.ConstexprBody

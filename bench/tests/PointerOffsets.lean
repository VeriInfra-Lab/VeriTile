import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.PointerOffsets
open VeriTile.Triton

def profile : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("n",.integer)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def source (X Y : RegionName) (xb yb n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [X := xb, Y := yb] in
  triton {
    offsets = tl.arange(0, B)
    mask = offsets < n
    x = tl.load(X + offsets, mask=mask)
    tl.store(Y + offsets, x, mask=mask)
  }

def expanded (X Y : RegionName) (xb yb n B : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  mask = offsets < $(n)
  x = tl.load(X + $(xb) + offsets, mask=mask)
  tl.store(Y + $(yb) + offsets, x, mask=mask)
}

theorem complete_source (X Y : RegionName) (xb yb n B : Nat) :
    source X Y xb yb n B = expanded X Y xb yb n B := rfl

/-- error: duplicate pointer offset binding -/
#guard_msgs in
def duplicate (X Y : RegionName) (xb yb n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [X := xb, X := yb] in
  triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x)
  }

/-- error: offset binding is not a pointer parameter -/
#guard_msgs in
def wrongRole (X Y : RegionName) (xb n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [n := xb] in
  triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x)
  }

/-- error: pointer offset must have a Nat element-offset carrier -/
#guard_msgs in
def signedOffset (X Y : RegionName) (xb : Int) (n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [X := xb] in
  triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x)
  }

/-- error: pointer-offset specialization does not yet support reassigned pointer arguments -/
#guard_msgs in
def mutated (X Y : RegionName) (xb n B : Nat) : ComputeKernel :=
  with_triton_call profile call_args [n := n] pointer_offsets [X := xb] in
  triton {
    X = X + 1
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    tl.store(Y + offsets, x)
  }

#axiomsClean complete_source
end VeriTile.Tests.PointerOffsets

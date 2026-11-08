import VeriTile.Triton.DSL.StaticSignedStrides
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StaticSignedStrides
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("N",.integer)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (B : Nat) (stop : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := stop] static_strides in triton {
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), B, tl.int64)
    for off in range(zero, N, step):
        value = tl.load(X + off)
        tl.store(Y + off, value)
  }

/-- Complete typed-program equality. The signed step register and counter
remain; the loop stride is exactly its immutable nonnegative constructor. -/
theorem complete_program (X Y : RegionName) (B : Nat) (stop : Int) :
    copied X Y B stop = (with_triton_call signature call_args [N := stop] in triton {
      N = tl.cast(N, tl.int64)
      zero = tl.full((), 0, tl.int64)
      step = tl.full((), tl.cast(B, tl.int64), tl.int64)
      for off in range(tl.cast(zero, tl.int64), tl.cast(N, tl.int64), $(B)) {
        value = tl.load(X + off)
        tl.store(Y + off, value)
      }
    }) := rfl

/-- error: static_strides requires immutable stride and constexpr bindings -/
#guard_msgs in
def reassigned (X Y : RegionName) (B : Nat) (stop : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := stop] static_strides in triton {
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), B, tl.int64)
    step += 1
    for off in range(zero, N, step):
        value = tl.load(X + off)
        tl.store(Y + off, value)
  }

/-- error: static_strides requires immutable stride and constexpr bindings -/
#guard_msgs in
def changedConstant (X Y : RegionName) (B : Nat) (stop : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := stop] static_strides in triton {
    B: tl.constexpr = 2
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), B, tl.int64)
    for off in range(zero, N, step):
        value = tl.load(X + off)
        tl.store(Y + off, value)
  }

/-- error: static_strides requires a signed tl.full stride from a natural constexpr -/
#guard_msgs in
def runtimeStride (X Y : RegionName) (B : Nat) (stop : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := stop] static_strides in triton {
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), N, tl.int64)
    for off in range(zero, N, step):
        value = tl.load(X + off)
        tl.store(Y + off, value)
  }

/-- error: static signed stride definition must dominate its range -/
#guard_msgs in
def beforeDefinition (X Y : RegionName) (B : Nat) (stop : Int) : ComputeKernel :=
  with_triton_call signature call_args [N := stop] static_strides in triton {
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    for off in range(zero, N, step):
        value = tl.load(X + off)
        tl.store(Y + off, value)
    step = tl.full((), B, tl.int64)
  }

#axiomsClean complete_program
end VeriTile.Tests.StaticSignedStrides

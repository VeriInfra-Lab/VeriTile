import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.LoopBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16),("Z",.fp32)]
  scalars := [("count",.integer)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y","Z"]

def source (X Y Z : RegionName) (count B : Nat) : ComputeKernel :=
  with_triton_call signature call_args [count := count] in
  triton {
    acc = tl.zeros([B], dtype=tl.float32)
    for off in range(0, count, B):
        offsets = off + tl.arange(0, B)
        x = tl.load(X + offsets).to(tl.float32)
        acc += x
        tl.store(Y + offsets, acc)
    tl.store(Z + tl.arange(0, B), acc)
  }

def expanded (X Y Z : RegionName) (count B : Nat) : ComputeKernel := triton {
  tl.ports inputs=[X], outputs=[Y,Z]
  tl.region X = tl.float32, Y = tl.float16, Z = tl.float32
  acc = tl.zeros([B], dtype=tl.float32)
  for off in range(0, $((count : Nat)), $(B)) {
    offsets = off + tl.arange(0, B)
    x = tl.load(X + offsets).to(tl.float32)
    acc += x
    tl.store(Y + offsets, acc.to(tl.float16))
  }
  tl.store(Z + tl.arange(0, B), acc)
}

/-- Narrowing occurs inside every iteration, while the carried FP32 value is
also available when the loop executes zero times. -/
theorem complete_loop (X Y Z : RegionName) (count B : Nat) :
    source X Y Z count B = expanded X Y Z count B := rfl

def overwrittenBound (X : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32)], constexpr := ["N"]} : CallSignature) in
  triton {
    N = 1
    for i in range(0, N, 1):
        tl.store(X + i, 0.0)
  }

/-- A constexpr parameter overwritten in the body is a local bound. The caller's
original N must never be substituted into this loop. -/
theorem overwritten_bound (X : RegionName) (N : Nat) :
    overwrittenBound X N = (triton {
      tl.region X = tl.float32
      N = 1
      for i in range((0), N, 1) {
        tl.store(X + i, (0.0).to(tl.float32))
      }
    } : ComputeKernel) := rfl

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def changingCarry (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp16)]} : CallSignature) in
  triton {
    acc = tl.zeros([4], dtype=tl.float32)
    for i in range(3):
        acc = tl.load(X)
  }

/-- error: unsupported statement involving a loop-carried source dtype -/
#guard_msgs in
def hiddenCarryWrite (X : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32)]} : CallSignature) in
  triton {
    acc = tl.zeros([4], dtype=tl.float32)
    for i in range(3):
        acc, other = tl.load(X), 0
  }

#axiomsClean complete_loop
#axiomsClean overwritten_bound

namespace ImmutableTwoArgumentRange

def source (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in triton {
    total = 0
    for _ in range(0, N):
        total += _
  }
/-- Immutable call-time endpoints choose a static natural loop. The original
underscore is both the written counter and the value read in the body. -/
theorem complete_program (N : Nat) : source N = triton {
    total = 0
    for «_» in range($(0), $(N), $(1)) {
      total += «_»
    }
  } := rfl

def overwritten (N : Nat) : ComputeKernel :=
  with_triton_signature ({constexpr := ["N"]} : CallSignature) in triton {
    N = 1
    for _ in range(0, N):
        total = _
  }
/-- Reassignment prevents caller N from becoming the actual loop endpoint. -/
theorem local_endpoint (N : Nat) : overwritten N = triton {
    N = 1
    for «_» in range((0), N, 1) {
      total = «_»
    }
  } := rfl

#axiomsClean complete_program
#axiomsClean local_endpoint
end ImmutableTwoArgumentRange

end VeriTile.Tests.LoopBody

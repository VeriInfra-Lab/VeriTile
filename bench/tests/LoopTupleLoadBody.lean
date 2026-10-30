import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.LoopTupleLoadBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["N"]

def source (X Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    y = -tl.load(Y)
    for i in range(N):
        next_x = x + y
        next_y = x - y
        x, y = next_x, next_y
    tl.store(X,x)
    tl.store(Y,y)
  }

def expected (X Y : RegionName) (N : Nat) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  x = tl.load(X)
  y = -tl.load(Y)
  tl.for i in $(N) {
    next_x = x + y
    next_y = x - y
    x, y = next_x, next_y
  }
  tl.store(X,x)
  tl.store(Y,y)
}

theorem preserves_load_negation_and_loop (X Y : RegionName) (N : Nat) :
    source X Y N = expected X Y N := rfl

def nestedArithmetic (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in triton {
    x = -tl.load(X) + 1.0
    tl.store(Y,x)
  }

def nestedArithmeticReference (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in triton {
    __veritile_source_load_0 = tl.load(X)
    x = -__veritile_source_load_0 + 1.0
    tl.store(Y,x)
  }

theorem nested_arithmetic_load_order (X Y : RegionName) :
    nestedArithmetic X Y = nestedArithmeticReference X Y := rfl
#axiomsClean nested_arithmetic_load_order

/-- error: loop-carried value requires a stable, known source dtype -/
#guard_msgs in
def tupleChangesType (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp16)]} : CallSignature) in triton {
    x = tl.load(X)
    y = tl.load(X)
    half = tl.load(Y)
    for i in range(3):
        x, y = half, half
  }

/-- error: tuple assignment needs unsupported simultaneous register capture -/
#guard_msgs in
def overwrittenTuple (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({pointers := [("X",.fp32),("Y",.fp32)]} : CallSignature) in triton {
    x = tl.load(X)
    y = tl.load(Y)
    for i in range(3):
        x, y = y, x
  }

#axiomsClean preserves_load_negation_and_loop
end VeriTile.Tests.LoopTupleLoadBody

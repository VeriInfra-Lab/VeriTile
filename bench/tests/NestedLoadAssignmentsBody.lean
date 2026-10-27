import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.NestedLoadAssignmentsBody
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X", .fp32), ("Y", .fp32)]
  scalars := [("scale", .fp32)]
  constexpr := ["FLAG"]

def source (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X).to(tl.float32) * scale
    if FLAG:
        y = tl.load(Y) + x
    else:
        y = tl.load(X) + x
    tl.store(Y, y)
  }

def reference (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    __veritile_source_load_0 = tl.load(X)
    x = __veritile_source_load_0.to(tl.float32) * scale
    if FLAG:
        __veritile_source_load_1 = tl.load(Y)
        y = __veritile_source_load_1 + x
    else:
        __veritile_source_load_2 = tl.load(X)
        y = __veritile_source_load_2 + x
    tl.store(Y, y)
  }

/-- The explicit cast and both branch-local loads retain their exact sites. -/
theorem complete (X Y : RegionName) (FLAG : Bool) :
    source X Y FLAG = reference X Y FLAG := by cases FLAG <;> rfl

/-- The actual first load remains a compute load, not an algebraic memory read. -/
theorem first_load (X Y : RegionName) (FLAG : Bool) :
    (source X Y FLAG).surfaceBody.head? = some
      (.assign .real [] "__veritile_source_load_0"
        (.compute (.load .fp32 (.region X (.constNat 0)) .none .zero))) := by
  cases FLAG <;> rfl

def collision (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    __veritile_source_load_0 = tl.load(Y)
    x = tl.load(X) * scale
    tl.store(Y, x)
  }

def collisionReference (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    __veritile_source_load_0 = tl.load(Y)
    __veritile_source_load_1 = tl.load(X)
    x = __veritile_source_load_1 * scale
    tl.store(Y, x)
  }

theorem fresh_names (X Y : RegionName) (FLAG : Bool) :
    collision X Y FLAG = collisionReference X Y FLAG := rfl

/-- error: nested floating load requires supported eager arithmetic operands -/
#guard_msgs in
def conditionalOperand (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = (tl.load(X) if FLAG else 0.0) * scale
    tl.store(Y, x)
  }

def integerOperand (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    i = tl.program_id(0)
    x = tl.load(X) * (i).to(tl.int64)
    tl.store(Y, x)
  }

def integerOperandReference (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    i = tl.program_id(0)
    __veritile_source_load_0 = tl.load(X)
    x = __veritile_source_load_0 * (i).to(tl.int64)
    tl.store(Y, x)
  }

/-- A cast on the other arithmetic operand is not a cast of the load. -/
theorem unrelated_integer_cast (X Y : RegionName) (FLAG : Bool) :
    integerOperand X Y FLAG = integerOperandReference X Y FLAG := rfl

/-- error: nested load source precision is unsupported by the current frontend -/
#guard_msgs in
def nestedIntegerLoad (X Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X).to(tl.int64) * scale
    tl.store(Y, x)
  }

#axiomsClean unrelated_integer_cast
#axiomsClean complete
#axiomsClean first_load
#axiomsClean fresh_names
end VeriTile.Tests.NestedLoadAssignmentsBody

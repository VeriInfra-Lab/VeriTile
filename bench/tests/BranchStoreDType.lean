import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.BranchStoreDType
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("gate",.fp32)]

def source (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    if gate > 0.0:
        y = x + 1.0
    else:
        y = x * 2.0
    tl.store(Y, y)
  }

def reference (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  tl.param gate : tl.float32
  x = tl.load(X)
  if gate > 0.0 {
    y = x + 1.0
  } else {
    y = x * 2.0
  }
  tl.store(Y, y)
}

/-- Both arms produce FP32. The complete kernel has no additional final cast. -/
theorem complete_fp32 (X Y : RegionName) : source X Y = reference X Y := rfl

def halfSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp16)]
  scalars := [("gate",.fp32)]

def narrow (X Y : RegionName) : ComputeKernel :=
  with_triton_signature halfSignature in triton {
    x = tl.load(X)
    if gate > 0.0:
        y = x + 1.0
    else:
        y = x * 2.0
    tl.store(Y, y)
  }

def narrowReference (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float16
  tl.param gate : tl.float32
  x = tl.load(X)
  if gate > 0.0 {
    y = x + 1.0
  } else {
    y = x * 2.0
  }
  tl.store(Y, y.to(tl.float16))
}

/-- Joining the branches must preserve the actual FP16 store narrowing. -/
theorem complete_fp16 (X Y : RegionName) : narrow X Y = narrowReference X Y := rfl

def oneArm (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.load(X)
    if gate > 0.0:
        x = x + 1.0
    tl.store(Y, x)
  }

def oneArmReference (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  tl.param gate : tl.float32
  x = tl.load(X)
  if gate > 0.0 {
    x = x + 1.0
  }
  tl.store(Y, x)
}

/-- A missing else joins with the unchanged entry type, including zero trips. -/
theorem complete_one_arm (X Y : RegionName) : oneArm X Y = oneArmReference X Y := rfl

def integerSignature : CallSignature where
  pointers := [("X",.i32),("Y",.i64)]

def functionCast (X Y : RegionName) : ComputeKernel :=
  with_triton_signature integerSignature in triton {
    x = tl.cast(tl.load(X), tl.int64)
    tl.store(Y, x)
  }

def methodCast (X Y : RegionName) : ComputeKernel :=
  with_triton_signature integerSignature in triton {
    x = tl.load(X).to(tl.int64)
    tl.store(Y, x)
  }

/-- Both source spellings retain the actual signed metadata read and widening. -/
theorem complete_integer_load (X Y : RegionName) : functionCast X Y = methodCast X Y := rfl

/-- error: nested load source precision is unsupported by the current frontend -/
#guard_msgs in
def floatingNested (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature in triton {
    x = tl.cast(tl.load(X), tl.float32)
    tl.store(Y, x)
  }

#axiomsClean complete_fp32
#axiomsClean complete_fp16
#axiomsClean complete_one_arm
#axiomsClean complete_integer_load
end VeriTile.Tests.BranchStoreDType

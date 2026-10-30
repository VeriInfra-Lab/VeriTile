import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ConstexprMaskBody
open VeriTile.Triton

def aliases (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({ pointers := [("X",.fp32),("Y",.fp32)] } : CallSignature) in triton {
    mask = None
    mask_copy = mask
    x = tl.load(X, mask_copy)
    tl.store(Y, x, mask=mask)
  }

def unmasked (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  x = tl.load(X)
  tl.store(Y,x)
}

theorem aliases_are_unmasked (X Y : RegionName) : aliases X Y = unmasked X Y := rfl

def selected (X Y : RegionName) (B : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature ({ pointers := [("X",.fp32),("Y",.fp32)], constexpr := ["B","FLAG"] } : CallSignature) in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    if FLAG:
        mask = None
    else:
        mask = offsets < B
    tl.store(Y + offsets, x, mask)
  }

def selectedView (X Y : RegionName) (B : Nat) (FLAG : Bool) : ComputeKernel :=
  match FLAG with
  | Bool.false => triton {
      tl.region X = tl.float32, Y = tl.float32
      offsets = tl.arange(0,$(B))
      x = tl.load(X + offsets)
      mask = offsets < $(B)
      tl.store(Y + offsets,x,mask)
    }
  | Bool.true => triton {
      tl.region X = tl.float32, Y = tl.float32
      offsets = tl.arange(0,$(B))
      x = tl.load(X + offsets)
      tl.store(Y + offsets,x)
    }

theorem both_cases (X Y : RegionName) (B : Nat) (FLAG : Bool) :
    selected X Y B FLAG = selectedView X Y B FLAG := by cases FLAG <;> rfl

def loopLocal (X Y : RegionName) : ComputeKernel :=
  with_triton_signature ({ pointers := [("X",.fp32),("Y",.fp32)] } : CallSignature) in triton {
    for i in range(2):
        mask = None
        mask_copy = mask
        x = tl.load(X, mask=mask_copy)
        tl.store(Y, x, mask)
  }

def loopView (X Y : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  tl.for i in 2 {
    x = tl.load(X)
    tl.store(Y,x)
  }
}

theorem loop_aliases (X Y : RegionName) : loopLocal X Y = loopView X Y := rfl

/-- error: None alias is only supported as a direct load/store mask -/
#guard_msgs in
def noneArithmetic (Y : RegionName) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.fp32)] } : CallSignature) in triton {
    mask = None
    x = mask + 1
    tl.store(Y,x)
  }

/-- error: None mask branches require an immutable Boolean constexpr test -/
#guard_msgs in
def runtimeBranch (Y : RegionName) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.fp32)] } : CallSignature) in triton {
    if tl.program_id(0) == 0:
        mask = None
    else:
        mask = True
    tl.store(Y,0.0,mask)
  }

/-- error: Boolean constexpr cannot be rebound during mask specialization -/
#guard_msgs in
def reboundFlag (Y : RegionName) (FLAG : Bool) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.fp32)], constexpr := ["FLAG"] } : CallSignature) in triton {
    FLAG = False
    if FLAG:
        mask = None
    else:
        mask = True
    tl.store(Y,0.0,mask)
  }

/-- error: possibly None or undefined mask requires assignment before use -/
#guard_msgs in
def zeroTripEscape (Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.fp32)], constexpr := ["N"] } : CallSignature) in triton {
    mask = True
    for i in range(N):
        mask = None
    tl.store(Y,0.0,mask)
  }

/-- error: possibly None or undefined mask requires assignment before use -/
#guard_msgs in
def previousIteration (Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.fp32)], constexpr := ["N"] } : CallSignature) in triton {
    mask = True
    for i in range(N):
        tl.store(Y,0.0,mask)
        mask = None
  }

/-- error: loop cannot change an incoming None alias -/
#guard_msgs in
def changedIncoming (Y : RegionName) (N : Nat) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.fp32)], constexpr := ["N"] } : CallSignature) in triton {
    mask = None
    for i in range(N):
        tl.store(Y,0.0,mask)
        mask = True
  }

/-- error: duplicate load/store mask -/
#guard_msgs in
def duplicateMask (Y : RegionName) : ComputeKernel :=
  with_triton_signature ({ pointers := [("Y",.fp32)] } : CallSignature) in triton {
    mask = None
    tl.store(Y,0.0,mask,mask=True)
  }

#axiomsClean aliases_are_unmasked
#axiomsClean both_cases
#axiomsClean loop_aliases
end VeriTile.Tests.ConstexprMaskBody

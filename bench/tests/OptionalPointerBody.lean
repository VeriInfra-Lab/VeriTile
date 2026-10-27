import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.OptionalPointerBody
open VeriTile.Triton

/-- The optional-pointer infix must not reserve an ordinary Lean identifier. -/
theorem ordinary_lean_names (is : Nat) :
    (fun is : Nat => is + 1) is = is + 1 := rfl

#axiomsClean ordinary_lean_names

def signature : CallSignature where
  pointers := [("X", .fp32), ("M", .fp32), ("Y", .fp32)]
  optionalPointers := ["M"]
  constexpr := ["B"]

def source (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [M := present] in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    if M is not None:
        mask = tl.load(M + offsets)
        x = x + mask
    tl.store(Y + offsets, x)
  }

def expanded (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel := triton {
  tl.region X = tl.float32, M = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  x = tl.load(X + offsets)
  if $((present : Bool)) {
    mask = tl.load(M + offsets)
    x = x + mask
  }
  tl.store(Y + offsets, x)
}

theorem guarded_load (X M Y : RegionName) (B : Nat) (present : Bool) :
    source X M Y B present = expanded X M Y B present := rfl

def inverseTest (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [M := present] in triton {
    offsets = tl.arange(0, B)
    x = tl.load(X + offsets)
    if M is None:
        x = x + x
    else:
        mask = tl.load(M + offsets)
        x = x + mask
    tl.store(Y + offsets, x)
  }

def inverseExpanded (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel := triton {
  tl.region X = tl.float32, M = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  x = tl.load(X + offsets)
  if not $((present : Bool)) {
    x = x + x
  } else {
    mask = tl.load(M + offsets)
    x = x + mask
  }
  tl.store(Y + offsets, x)
}

theorem inverse_branch (X M Y : RegionName) (B : Nat) (present : Bool) :
    inverseTest X M Y B present = inverseExpanded X M Y B present := rfl

def nested (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [M := present] in triton {
    offsets = tl.arange(0, B)
    for i in range(2):
        if M is not None:
            mask = tl.load(M + offsets)
            tl.store(Y + offsets, mask)
  }

def nestedExpanded (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel := triton {
  tl.region X = tl.float32, M = tl.float32, Y = tl.float32
  offsets = tl.arange(0, $(B))
  tl.for i in 2 {
    if $((present : Bool)) {
      mask = tl.load(M + offsets)
      tl.store(Y + offsets, mask)
    }
  }
}

theorem nested_guard (X M Y : RegionName) (B : Nat) (present : Bool) :
    nested X M Y B present = nestedExpanded X M Y B present := rfl

/-- error: every optional pointer needs an explicit caller presence binding -/
#guard_msgs in
def missing (X M Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature in triton {
    if M is not None:
        tl.store(Y, tl.load(M))
  }

/-- error: optional pointer use requires its non-None branch -/
#guard_msgs in
def unguarded (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [M := present] in triton {
    tl.store(Y, tl.load(M))
  }

/-- error: optional pointer use requires its non-None branch -/
#guard_msgs in
def absent_branch (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [M := present] in triton {
    if M is None:
        tl.store(Y, tl.load(M))
  }

/-- error: duplicate pointer presence binding -/
#guard_msgs in
def duplicate (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [M := present, M := present] in triton {
    tl.store(Y, 0)
  }

/-- error: presence binding requires a declared optional pointer -/
#guard_msgs in
def unknown (X M Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [X := present, M := present] in triton {
    tl.store(Y, 0)
  }

#axiomsClean guarded_load
#axiomsClean inverse_branch
#axiomsClean nested_guard
end VeriTile.Tests.OptionalPointerBody

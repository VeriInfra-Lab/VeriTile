import VeriTile.Triton.DSL.OptionalPointers
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.OptionalPointers
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := []
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature optional_pointers [X := present] in triton {
    if X is not None:
        values = tl.load(X + tl.arange(0, B))
        tl.store(Y + tl.arange(0, B), values)
  }

/-- Both presence values preserve the complete typed conditional program. -/
theorem complete_program (X Y : RegionName) (B : Nat) (present : Bool) :
    copied X Y B present = (with_triton_signature signature in triton {
      if present {
        values = tl.load(X + tl.arange(0, B))
        tl.store(Y + tl.arange(0, B), values)
      }
    }) := by cases present <;> rfl

def reverseGuard (X Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature optional_pointers [X := present] in triton {
    if X is None:
        value = 0.0
    else:
        values = tl.load(X + tl.arange(0, B))
        tl.store(Y + tl.arange(0, B), values)
  }

theorem absent_guard (X Y : RegionName) (B : Nat) (present : Bool) :
    reverseGuard X Y B present = (with_triton_signature signature in triton {
      if not present {
        value = 0.0
      } else {
        values = tl.load(X + tl.arange(0, B))
        tl.store(Y + tl.arange(0, B), values)
      }
    }) := by cases present <;> rfl

def capture (X Y : RegionName) (B : Nat) (present : Bool) : ComputeKernel :=
  with_triton_signature signature optional_pointers [X := present] in triton {
    present: tl.constexpr = 0
    if X is not None:
        values = tl.load(X + tl.arange(0, B))
        tl.store(Y + tl.arange(0, B), values)
  }

theorem caller_presence (X Y : RegionName) (B : Nat) (present : Bool) :
    capture X Y B present = copied X Y B present := by cases present <;> rfl

/-- error: optional pointer use requires a dominating presence guard -/
#guard_msgs in
def unguarded (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature optional_pointers [X := Bool.false] in triton {
    value = tl.load(X + 0)
    tl.store(Y + 0, value)
  }

/-- error: optional pointer use requires a dominating presence guard -/
#guard_msgs in
def wrongBranch (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature optional_pointers [X := Bool.false] in triton {
    if X is None:
        value = tl.load(X + 0)
        tl.store(Y + 0, value)
  }

/-- error: optional_pointers currently requires an immutable pointer parameter -/
#guard_msgs in
def mutation (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature optional_pointers [X := Bool.true] in triton {
    X = Y
    if X is not None:
        value = tl.load(X + 0)
  }

#axiomsClean complete_program
#axiomsClean absent_guard
#axiomsClean caller_presence
end VeriTile.Tests.OptionalPointers

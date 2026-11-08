import VeriTile.Triton.DSL.PointerCells
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PointerCells
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (x0 y0 width B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_cells [X := (x0,width), Y := (y0,width)] in triton {
    offsets = tl.arange(0, B)
    values = tl.load(X + offsets)
    tl.store(Y + offsets, values)
  }

/-- Equality of the complete typed program for every base, cell width, shape
and aliasing choice. The full displacement is scaled once, outside the source. -/
theorem complete_program (X Y : RegionName) (x0 y0 width B : Nat) :
    copied X Y x0 y0 width B = (with_triton_signature signature in triton {
      offsets = tl.arange(0, B)
      values = tl.load(X + ($(x0) + $(width) * offsets))
      tl.store(Y + ($(y0) + $(width) * offsets), values)
    }) := rfl

def capture (X Y : RegionName) (width x0 B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_cells [X := (x0,width)] in triton {
    width: tl.constexpr = 0
    x0: tl.constexpr = 0
    offsets = tl.arange(0, B)
    values = tl.load(X + offsets)
    tl.store(Y + offsets, values)
  }

/-- Pasted constexpr declarations cannot capture either caller-side coordinate. -/
theorem caller_coordinates (X Y : RegionName) (width x0 B : Nat) :
    capture X Y width x0 B = (with_triton_signature signature in triton {
      offsets = tl.arange(0, B)
      values = tl.load(X + ($(x0) + $(width) * offsets))
      tl.store(Y + offsets, values)
    }) := rfl

/-- error: duplicate pointer cell-view binding -/
#guard_msgs in
def duplicate (X Y : RegionName) (B : Nat) : ComputeKernel :=
  with_triton_signature signature pointer_cells [X := (0,1), X := (0,2)] in triton {
    values = tl.load(X + tl.arange(0, B))
    tl.store(Y + tl.arange(0, B), values)
  }

/-- error: pointer_cells currently requires an immutable pointer parameter -/
#guard_msgs in
def mutation (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature pointer_cells [X := (0,4)] in triton {
    X += 1
    values = tl.load(X + 0)
    tl.store(Y + 0, values)
  }

/-- error: pointer_cells requires a direct pointer plus a complete natural displacement -/
#guard_msgs in
def escaping (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature pointer_cells [X := (0,4)] in triton {
    P = X
    values = tl.load(P + 0)
    tl.store(Y + 0, values)
  }

/-- error: pointer_cells requires one parenthesized complete displacement -/
#guard_msgs in
def chained (X Y : RegionName) : ComputeKernel :=
  with_triton_signature signature pointer_cells [X := (0,4)] in triton {
    values = tl.load(X + 1 + 2)
    tl.store(Y + 0, values)
  }

#axiomsClean complete_program
#axiomsClean caller_coordinates
end VeriTile.Tests.PointerCells

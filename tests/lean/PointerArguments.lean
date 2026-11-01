import VeriTile.Triton.DSL.PointerArguments
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.PointerArguments
open VeriTile.Triton FP.Structural

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("stride",.i64)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (x0 y0 B : Nat) (stride : Int) : ComputeKernel :=
  with_triton_call signature call_args [stride := stride]
    pointer_offsets [X := x0, Y := y0] in triton {
    offsets = tl.arange(0, B).to(tl.int64)
    values = tl.load(X + offsets * stride)
    tl.store(Y + offsets, values)
  }

/-- All pointer identities, aliases, offsets, shapes and signed strides are
retained in the complete typed program. -/
theorem complete_program (X Y : RegionName) (x0 y0 B : Nat) (stride : Int) :
    copied X Y x0 y0 B stride =
      (with_triton_call signature call_args [stride := stride] in triton {
        offsets = tl.arange(0, B).to(tl.int64)
        values = tl.load((X + $(x0)) + offsets * stride)
        tl.store((Y + $(y0)) + offsets, values)
      }) := rfl

def scalarSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  readPorts := ["X"]
  writePorts := ["Y"]

def negative (X Y : RegionName) : ComputeKernel :=
  with_triton_signature scalarSignature pointer_offsets [X := 1, Y := 0] in triton {
    value = tl.load(X - 1)
    tl.store(Y, value)
  }

/-- A negative displacement is added to the supplied base before conversion.
Clamping the displacement alone would incorrectly read X[1]. -/
theorem negative_address {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (s : State α) :
    ∃ t, exec M (negative X Y) s=some t ∧
      t.mem Y 0=.mk .real (M.fp32Load ((s.mem X 0).read .real)) := by
  refine ⟨?_,?_,?_⟩
  rotate_left
  · simp [negative,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,bop,Region.cast,store]
    rfl
  · simp [State.write,numeric]

def capture (X Y : RegionName) (off : Nat) : ComputeKernel :=
  with_triton_signature scalarSignature pointer_offsets [X := off] in triton {
    off: tl.constexpr = 0
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- A source constexpr with the same name cannot replace the caller offset. -/
theorem caller_offset (X Y : RegionName) (off : Nat) : capture X Y off =
    (with_triton_signature scalarSignature in triton {
      value = tl.load(X + $(off))
      tl.store(Y, value)
    }) := rfl

def matrixSignature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  constexpr := ["M","N"]
  readPorts := ["X"]
  writePorts := ["Y"]

def sliced (X Y : RegionName) (x0 y0 M N : Nat) : ComputeKernel :=
  with_triton_signature matrixSignature pointer_offsets [X := x0, Y := y0] in triton {
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    values = tl.load(X + rows * N + cols)
    tl.store(Y + rows * N + cols, values)
  }

theorem complete_slices (X Y : RegionName) (x0 y0 M N : Nat) : sliced X Y x0 y0 M N =
    (with_triton_signature matrixSignature in triton {
      rows = tl.arange(0, M)[:, None]
      cols = tl.arange(0, N)[None, :]
      values = tl.load((X + $(x0)) + rows * N + cols)
      tl.store((Y + $(y0)) + rows * N + cols, values)
    }) := rfl

/-- error: duplicate pointer offset binding -/
#guard_msgs in
def duplicate (X Y : RegionName) : ComputeKernel :=
  with_triton_signature scalarSignature pointer_offsets [X := 1, X := 2] in triton {
    value = tl.load(X)
    tl.store(Y, value)
  }

/-- error: pointer_offsets currently requires an immutable pointer parameter -/
#guard_msgs in
def changedPointer (X Y : RegionName) : ComputeKernel :=
  with_triton_signature scalarSignature pointer_offsets [X := 1] in triton {
    X += 1
    value = tl.load(X)
    tl.store(Y, value)
  }

#axiomsClean complete_program
#axiomsClean negative_address
#axiomsClean caller_offset
#axiomsClean complete_slices
end VeriTile.Tests.PointerArguments

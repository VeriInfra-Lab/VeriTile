import VeriTile.Triton.DSL.GuardedOptionalRebinding
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.GuardedOptionalRebinding
open VeriTile.Triton FP.Structural

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("stride",.integer)]
  constexpr := ["B"]
  readPorts := ["X"]
  writePorts := ["Y"]

def copied (X Y : RegionName) (B stride : Nat) (present : Bool) : ComputeKernel :=
  with_triton_call signature call_args [stride := stride] guarded_optional_pointers [X := present] in triton {
    if X is not None:
        X = X + stride + tl.arange(0, B)
        values = tl.load(X).to(tl.float32)
        tl.store(Y + tl.arange(0, B), values.to(Y.dtype.element_ty))
  }

def expected (X Y : RegionName) (B stride : Nat) (present : Bool) : ComputeKernel :=
  with_triton_call signature call_args [stride := stride] in triton {
    if present {
      __veritile_optional_X = X + stride + tl.arange(0, B)
      values = tl.load(__veritile_optional_X).to(tl.float32)
      tl.store(Y + tl.arange(0, B), values.to(Y.dtype.element_ty))
    }
  }

/-- All dimensions, offsets and both presence values retain the complete program. -/
theorem full_program (X Y : RegionName) (B stride : Nat) (present : Bool) :
    copied X Y B stride present = expected X Y B stride present := by
  cases present <;> rfl

theorem full_execution {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (B stride : Nat) (present : Bool) (entry : State α) :
    exec M (copied X Y B stride present) entry = exec M (expected X Y B stride present) entry := by
  rw [full_program]

theorem full_safety {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (B stride : Nat) (present : Bool) (entry : State α) :
    TraceSafe M bounds (copied X Y B stride present) entry ↔
      TraceSafe M bounds (expected X Y B stride present) entry := by
  rw [full_program]

/-- An absent pointer performs no access and does not require a valid allocation. -/
theorem absent_execution {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (B stride : Nat) (entry : State α) :
    exec M (copied X Y B stride Bool.false) entry = some entry := by
  simp [copied,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold]

theorem absent_safety {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (X Y : RegionName) (B stride : Nat) (entry : State α) :
    TraceSafe M bounds (copied X Y B stride Bool.false) entry := by
  simp only [TraceSafe,copied,ComputeKernel.surfaceBody]
  simp [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,step]

/-- error: optional pointer use requires a dominating presence guard -/
#guard_msgs in
def unguarded (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.true] in triton {
    values = tl.load(X)
  }

/-- error: optional_pointers currently requires an immutable pointer parameter -/
#guard_msgs in
def wrong_branch (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.false] in triton {
    if X is None:
        X = X + 1
        values = tl.load(X)
  }

/-- error: guarded optional pointer permits exactly one rebase -/
#guard_msgs in
def repeated_rebase (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.true] in triton {
    if X is not None:
        X = X + 1
        X = X + 1
        values = tl.load(X)
  }

/-- error: guarded optional pointer permits exactly one rebase -/
#guard_msgs in
def nested_rebase (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.true] in triton {
    if X is not None:
        X = X + 1
        if True:
            X = X + 1
        values = tl.load(X)
  }

/-- error: a loop cannot shadow a guarded optional pointer -/
#guard_msgs in
def loop_shadow (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.true] in triton {
    if X is not None:
        X = X + 1
        for X in range(2):
            value = 0.0
        values = tl.load(X)
  }

/-- error: optional pointer offset cannot read the pointer being rebased -/
#guard_msgs in
def self_offset (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.true] in triton {
    if X is not None:
        X = X + X
        values = tl.load(X)
  }

/-- error: guarded optional pointer SSA name collision -/
#guard_msgs in
def name_collision (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.true] in triton {
    __veritile_optional_X = 0
    if X is not None:
        X = X + 1
        values = tl.load(X)
  }

/-- error: optional_pointers currently requires an immutable pointer parameter -/
#guard_msgs in
def other_pointer (X Y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [stride := 1] guarded_optional_pointers [X := Bool.true] in triton {
    if X is not None:
        X = Y + 1
        values = tl.load(X)
  }

#axiomsClean full_program
#axiomsClean full_execution
#axiomsClean full_safety
#axiomsClean absent_execution
#axiomsClean absent_safety
end VeriTile.Tests.GuardedOptionalRebinding

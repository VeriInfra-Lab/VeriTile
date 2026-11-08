import VeriTile.Triton.DSL.StaticAssertions
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.StaticAssertions
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

/-- These checks cover arbitrary complete bodies, including their failure
behavior and safety, rather than only the fixture's marker assignment. -/
theorem checked_execution {α : Type} [Inhabited α] (M : Algebra α)
    (kernel : ComputeKernel) (s : State α) :
    exec M (DSL.StaticAssertions.checked Bool.true kernel) s = exec M kernel s := by
  cases kernel
  simp [DSL.StaticAssertions.checked,exec,run,step,evalOp_unfold,Option.bind]

theorem checked_rejection {α : Type} [Inhabited α] (M : Algebra α)
    (kernel : ComputeKernel) (s : State α) :
    exec M (DSL.StaticAssertions.checked Bool.false kernel) s = none := by
  cases kernel
  simp [DSL.StaticAssertions.checked,exec,run,step,evalOp_unfold,Option.bind]

theorem checked_safety {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (kernel : ComputeKernel) (s : State α) :
    TraceSafe M bounds (DSL.StaticAssertions.checked Bool.true kernel) s ↔ TraceSafe M bounds kernel s := by
  cases kernel
  simp [TraceSafe,DSL.StaticAssertions.checked,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeOp_unfold,
    step,evalOp_unfold,Option.bind]

theorem checked_unsafe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (kernel : ComputeKernel) (s : State α) :
    ¬ TraceSafe M bounds (DSL.StaticAssertions.checked Bool.false kernel) s := by
  cases kernel
  simp [TraceSafe,DSL.StaticAssertions.checked,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeOp_unfold,
    step,evalOp_unfold,Option.bind]

def signature : CallSignature where
  constexpr := ["N", "FLAG"]

def source (N : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    tl.static_assert(FLAG and N == 128, "fixed source tile")
    marker = 7.0
  }

def signedSubtraction (N : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    tl.static_assert(N - 10 == -5)
    marker = 7.0
  }

/-- A satisfied assertion produces an actual successful complete execution,
with exactly the source's register write and no memory change. -/
theorem accepted_execution {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    exec M (source 128 Bool.true) s = some (s.setReg "marker" .real [] (fun _ => M.literal none .real 7.0)) := by
  simp [source,DSL.StaticAssertions.checked,exec,run,step,evalExpr,evalOp_unfold,Option.bind]

theorem rejected_execution {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    exec M (source 127 Bool.true) s = none := by
  simp [source,DSL.StaticAssertions.checked,exec,run,step,evalExpr,evalOp_unfold,Option.bind]

theorem boolean_rejection {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    exec M (source 128 Bool.false) s = none := by
  simp [source,DSL.StaticAssertions.checked,exec,run,step,evalExpr,evalOp_unfold,Option.bind]

/-- Python constexpr subtraction uses signed integers even for natural-sized
caller metadata; it must not silently truncate 5-10 to zero. -/
theorem signed_host_arithmetic {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    exec M (signedSubtraction 5 Bool.true) s = some (s.setReg "marker" .real [] (fun _ => M.literal none .real 7.0)) := by
  simp [signedSubtraction,DSL.StaticAssertions.checked,exec,run,step,evalExpr,evalOp_unfold,Option.bind]

theorem accepted_safety {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (s : State α) : TraceSafe M bounds (source 128 Bool.true) s := by
  simp [TraceSafe,source,DSL.StaticAssertions.checked,ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.traceSafeList_nil,
    Safety.stepSafe_unfold,Safety.safeAlg_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,
    step,evalExpr,evalOp_unfold,Option.bind]

theorem rejected_safety {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (s : State α) : ¬ TraceSafe M bounds (source 127 Bool.true) s := by
  simp [TraceSafe,source,DSL.StaticAssertions.checked,ComputeKernel.surfaceBody,Safety.traceSafeList_cons,Safety.traceSafeList_nil,
    Safety.stepSafe_unfold,Safety.safeAlg_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,
    step,evalExpr,evalOp_unfold,Option.bind]

def runtimeSignature : CallSignature where
  scalars := [("N",.integer)]

/-- error: static assertions may only read immutable integer/Boolean constexpr arguments -/
#guard_msgs in
def runtime_predicate (N : Nat) : ComputeKernel :=
  with_triton_call runtimeSignature call_args [N := N] in triton {
    tl.static_assert(N == 128)
    marker = 7.0
  }

/-- error: static assertions may only read immutable integer/Boolean constexpr arguments -/
#guard_msgs in
def floating_predicate (N : ℝ) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    tl.static_assert(N == 128)
    marker = 7.0
  }

/-- error: static assertions may only read immutable integer/Boolean constexpr arguments -/
#guard_msgs in
def rebound_metadata (N : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    tl.static_assert(N == 128)
    N = 128
  }

/-- error: static assertions must be top-level statements -/
#guard_msgs in
def nested_assertion (N : Nat) (FLAG : Bool) : ComputeKernel :=
  with_triton_call signature call_args [] in triton {
    if FLAG:
        tl.static_assert(N == 128)
    marker = 7.0
  }

#axiomsClean accepted_execution
#axiomsClean checked_execution
#axiomsClean checked_rejection
#axiomsClean checked_safety
#axiomsClean checked_unsafe
#axiomsClean rejected_execution
#axiomsClean boolean_rejection
#axiomsClean signed_host_arithmetic
#axiomsClean accepted_safety
#axiomsClean rejected_safety
end
end VeriTile.Tests.StaticAssertions

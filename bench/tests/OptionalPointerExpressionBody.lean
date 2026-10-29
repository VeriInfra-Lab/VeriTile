import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.OptionalPointerExpressionBody
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  pointers := [("Mean",.fp32)]
  optionalPointers := ["Mean"]

def source (Mean : RegionName) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [Mean := present] in triton {
    value = tl.load(Mean) if Mean is not None else 0.0
  }

def inverse (Mean : RegionName) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [Mean := present] in triton {
    value = 0.0 if Mean is None else tl.load(Mean)
  }

/-- An absent argument needs no allocated cell, even though the original
function body still visibly contains the guarded load. -/
theorem absent_needs_no_storage {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (Mean : RegionName) :
    FP.Structural.TraceSafe M (fun _ => 0) (source Mean Bool.false) s := by
  simp [FP.Structural.TraceSafe,source,ComputeKernel.surfaceBody,Safety.traceSafeList,
    Safety.stepSafe,Safety.safeAlg,Safety.safeOp.eq_def,Safety.safeExpr,
    Safety.safeCompute,Safety.safeAddress,Safety.safeMask,
    evalExpr,evalComputeOp,Region.cast,evalOp_unfold,run,step,Option.bind]

theorem inverse_absent_needs_no_storage {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (Mean : RegionName) :
    FP.Structural.TraceSafe M (fun _ => 0) (inverse Mean Bool.false) s := by
  simp [FP.Structural.TraceSafe,inverse,ComputeKernel.surfaceBody,Safety.traceSafeList,
    Safety.stepSafe,Safety.safeAlg,Safety.safeOp.eq_def,Safety.safeExpr,
    Safety.safeCompute,Safety.safeAddress,Safety.safeMask,
    evalExpr,evalComputeOp,Region.cast,evalOp_unfold,run,step,Option.bind]

/-- Presence retains the real load obligation; it cannot be replaced by the
fallback merely to accept a missing allocation. -/
theorem present_requires_storage {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (Mean : RegionName) :
    ¬ FP.Structural.TraceSafe M (fun _ => 0) (source Mean Bool.true) s := by
  simp [FP.Structural.TraceSafe,source,ComputeKernel.surfaceBody,Safety.traceSafeList,
    Safety.stepSafe,Safety.safeAlg,Safety.safeOp.eq_def,Safety.safeExpr,
    Safety.addressSafe,Safety.maskActive,Safety.safeCompute,Safety.safeAddress,Safety.safeMask,
    evalExpr,evalComputeOp,Region.cast,evalOp_unfold,run,step,Option.bind]

def nested (Mean : RegionName) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [Mean := present] in triton {
    for row in range(1):
        value = tl.load(Mean + row) if Mean is not None else 0.0
  }

theorem nested_absent_needs_no_storage {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (Mean : RegionName) :
    FP.Structural.TraceSafe M (fun _ => 0) (nested Mean Bool.false) s := by
  simp [FP.Structural.TraceSafe,nested,ComputeKernel.surfaceBody,Safety.traceSafeList,
    Safety.stepSafe,Safety.safeAlg,Safety.safeOp.eq_def,Safety.safeExpr,
    Safety.safeCompute,Safety.safeAddress,Safety.safeMask,Safety.loopSafe,Safety.rangeSafe,
    evalExpr,evalComputeOp,Region.cast,evalOp_unfold,run,step,loop,range,Option.bind]

#axiomsClean nested_absent_needs_no_storage

def correlatedSignature : CallSignature :=
  { signature with constexpr := ["RMS"] }

/-- A flag-guarded name is absent in the other specialization. Both loop
accumulations must elaborate without reading that uninitialized name. -/
def correlated (Mean : RegionName) (RMS : Bool) : ComputeKernel :=
  let present := !RMS
  with_triton_signature correlatedSignature pointer_presence [Mean := present] in triton {
    total = tl.zeros((1,), dtype=tl.float32)
    for row in range(1):
        if not RMS:
            mean = tl.load(Mean) if Mean is not None else 0.0
        xhat = mean if not RMS else 1.0
        total += xhat
  }

theorem rms_branch_needs_no_storage {α : Type} [Inhabited α]
    (M : Algebra α) (s : State α) (Mean : RegionName) :
    FP.Structural.TraceSafe M (fun _ => 0) (correlated Mean Bool.true) s := by
  simp [FP.Structural.TraceSafe,correlated,ComputeKernel.surfaceBody,Safety.traceSafeList,
    Safety.stepSafe,Safety.safeAlg,Safety.safeOp.eq_def,Safety.safeExpr,
    Safety.safeCompute,Safety.safeAddress,Safety.safeMask,Safety.loopSafe,Safety.rangeSafe,
    evalExpr,evalComputeOp,Region.cast,evalOp_unfold,run,step,loop,range,Option.bind,
    ComputeDType.eraseDType,State.setReg]

#axiomsClean rms_branch_needs_no_storage

/-- error: optional pointer use requires its non-None branch -/
#guard_msgs in
def wrong_branch (Mean : RegionName) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [Mean := present] in triton {
    value = tl.load(Mean) if Mean is None else 0.0
  }

/-- error: optional pointer use requires its non-None branch -/
#guard_msgs in
def unguarded_fallback (Mean : RegionName) (present : Bool) : ComputeKernel :=
  with_triton_signature signature pointer_presence [Mean := present] in triton {
    value = 0.0 if Mean is not None else tl.load(Mean)
  }

#axiomsClean absent_needs_no_storage
#axiomsClean inverse_absent_needs_no_storage
#axiomsClean present_requires_storage
end
end VeriTile.Tests.OptionalPointerExpressionBody

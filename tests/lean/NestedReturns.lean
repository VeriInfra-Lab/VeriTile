import VeriTile.Triton.DSL.NestedReturns
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.NestedReturns
open VeriTile.Triton
noncomputable section
def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("x",.fp32)]
  constexpr := ["ENABLED"]
  readPorts := []
  writePorts := ["Y"]
def pasted (Y : RegionName) (x : ℝ) (ENABLED : Bool) : ComputeKernel :=
  with_triton_call signature call_args [x := x] in triton {
    if ENABLED:
        if x.to(tl.int32) > 0:
            return
    tl.store(Y, 1.0)
  }
/-- An unavailable inner conversion is never evaluated when the outer guard
is false. The following store is still executed. -/
theorem disabled_executes_store {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (Y : RegionName) (x : ℝ) (s : FP.Structural.State α) :
    ∃ t, FP.Structural.exec M (pasted Y x Bool.false) s=some t ∧
      t.mem Y 0=.mk .real (M.cast (some .fp32) .real .real (M.literal (some .fp32) .real 1.0)) := by
  simp [pasted,FP.Structural.exec,ComputeKernel.surfaceBody,FP.Structural.run,
    FP.Structural.step,FP.Structural.evalOp_unfold,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.State.write,FP.Structural.ofFloat,FP.Structural.toFloat,
    TileShape.allIndices,TileShape.axisDim,FP.Structural.store,Option.bind]

def flagsSignature : CallSignature where
  pointers := [("Y",.fp32)]
  constexpr := ["ENABLED","BLOCKED"]
  readPorts := []
  writePorts := ["Y"]
def flags (Y : RegionName) (ENABLED BLOCKED : Bool) : ComputeKernel :=
  with_triton_call flagsSignature call_args [] in triton {
    if ENABLED:
        if BLOCKED:
            return
    tl.store(Y, 1.0)
  }
/-- A true nested return skips the complete function continuation. -/
theorem nested_return_skips_store {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (Y : RegionName) (s : FP.Structural.State α) :
    FP.Structural.exec M (flags Y Bool.true Bool.true) s=some s := by
  simp [flags,FP.Structural.exec,FP.Structural.run,FP.Structural.step,
    FP.Structural.evalOp_unfold,FP.Structural.bop]

/- Unsupported control flow is rejected before a nested return can be lost. -/
/-- error: unsupported conditional return: expected a chain of sole conditional statements -/
#guard_msgs in
def rejects_preceding_effect (Y : RegionName) (ENABLED BLOCKED : Bool) : ComputeKernel :=
  with_triton_call flagsSignature call_args [] in triton {
    if ENABLED:
        tl.store(Y, 0.0)
        if BLOCKED:
            return
    tl.store(Y, 1.0)
  }

/-- error: unsupported conditional return: expected a chain of sole conditional statements -/
#guard_msgs in
def rejects_return_from_loop (Y : RegionName) (ENABLED BLOCKED : Bool) : ComputeKernel :=
  with_triton_call flagsSignature call_args [] in triton {
    while ENABLED:
        if BLOCKED:
            return
    tl.store(Y, 1.0)
  }

#axiomsClean disabled_executes_store
#axiomsClean nested_return_skips_store
end
end VeriTile.Tests.NestedReturns

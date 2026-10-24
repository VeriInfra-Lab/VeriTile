import VeriTile.Triton.DSL.CallSignature
import VeriTile.Meta.StatementAudit
import VeriTile.Triton.Float.Structural

namespace VeriTile.Tests.FP32CallValue
open VeriTile.Triton

def signature : CallSignature where
  pointers := [("X",.fp32),("Y",.fp32)]
  scalars := [("gain",.fp32)]

def source (X Y : RegionName) (gain : ℝ) : ComputeKernel :=
  with_triton_call signature call_args [gain := gain] in
  triton {
    x = tl.load(X)
    y = x * gain
    tl.store(Y, y)
  }

def expanded (X Y : RegionName) (gain : ℝ) : ComputeKernel := triton {
  tl.region X = tl.float32, Y = tl.float32
  x = tl.load(X)
  y = x * $((gain : ℝ))
  tl.store(Y, y)
}

theorem fp32_context (X Y : RegionName) (gain : ℝ) :
    source X Y gain = expanded X Y gain := rfl

def float64Context (X Y : RegionName) (gain : ℝ) : ComputeKernel :=
  with_triton_call ({pointers := [("X",.fp64),("Y",.fp64)], scalars := [("gain",.fp32)]} : CallSignature)
    call_args [gain := gain] in
  triton {
    x = tl.load(X)
    y = x * gain
    tl.store(Y, y)
  }

def narrowContext (Y : RegionName) (gain : ℝ) : ComputeKernel :=
  with_triton_call ({pointers := [("Y",.fp16)], scalars := [("gain",.fp32)]} : CallSignature)
    call_args [gain := gain] in
  triton {
    tl.store(Y, gain)
  }

/-- error: per-call FP32 literal must have a Real carrier -/
#guard_msgs in
def integerCarrier (X Y : RegionName) (gain : Nat) : ComputeKernel :=
  with_triton_call signature call_args [gain := gain] in
  triton {
    x = tl.load(X)
    tl.store(Y, x * gain)
  }

open FP.Structural

/-- FP32 call values are rounded in their own source context before widening
into an FP64 tensor computation. The destination has no redundant store cast. -/
theorem widened_parameter {α : Type} [Inhabited α] (M : Algebra α)
    (X Y : RegionName) (gain : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec M (float64Context X Y gain) s = some t ∧
      t.regs .real [] "y" = some (fun _ =>
        M.binary (some .fp64) .real .mul ((s.mem X 0).read .real)
          (M.literal (some .fp32) .real gain)) := by
  simp [float64Context, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, store, State.setReg, ComputeDType.eraseDType, Region.cast, FP.Structural.bop, ofFloat, toFloat]
  all_goals rfl

/-- An implicit FP16 store conversion happens after the actual FP32 call value
is obtained, preserving its first rounding rather than reinterpreting it as FP16. -/
theorem narrowed_parameter {α : Type} [Inhabited α] (M : Algebra α)
    (Y : RegionName) (gain : ℝ) (s : State α) :
    ∃ t, FP.Structural.exec M (narrowContext Y gain) s = some t ∧
      t.mem Y 0 = Cell.mk .fp16
        (M.cast none .real .fp16 (M.literal (some .fp32) .real gain)) := by
  simp [narrowContext, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, numeric, store, ofFloat, toFloat]

#axiomsClean fp32_context
#axiomsClean widened_parameter
#axiomsClean narrowed_parameter
namespace Selection

def signature : CallSignature where
  pointers := [("Y",.fp32)]
  scalars := [("gain",.fp32)]
  constexpr := ["choose"]

def source (Y : RegionName) (gain : ℝ) (choose : Bool) : ComputeKernel :=
  with_triton_call signature call_args [gain := gain] in
  triton {
    chosen = tl.where(choose, -gain, gain)
    tl.store(Y, chosen)
  }

theorem selection_precision {α : Type} [Inhabited α] (M : Algebra α)
    (Y : RegionName) (gain : ℝ) (choose : Bool) (s : State α) :
    ∃ t, FP.Structural.exec M (source Y gain choose) s = some t ∧
      t.regs .real [] "chosen" = some (fun _ =>
        if choose then M.unary (some .fp32) (.neg .real) (M.literal (some .fp32) .real gain)
        else M.literal (some .fp32) .real gain) := by
  cases choose <;>
    simp [source, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
      evalOp_unfold, numeric, store, State.setReg, State.write, ComputeDType.eraseDType, Region.cast, FP.Structural.bop, ofFloat, toFloat]
  all_goals rfl
#axiomsClean selection_precision
end Selection
end VeriTile.Tests.FP32CallValue

import VeriTile.Triton.DSL.ScalarView
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.SourceScalarView
open VeriTile.Triton FP.Structural
noncomputable section

def source (X O : RegionName) (flag : Bool) : ComputeKernel := triton {
  tl.region X = tl.float32, O = tl.float32
  tl.param scale : tl.float32
  x = tl.load(X)
  if flag {
    y = x * scale
  } else {
    y = scale + x
  }
  tl.store(O, y)
}

def view (X O : RegionName) (flag : Bool) (epsilon : ℝ) : ComputeKernel :=
  source_scalar_view (source X O flag) at "scale" := epsilon

/-- A scalar view matches complete execution, including failure, registers
and memory, when the actual scalar ABI register has the specified value. -/
theorem execution {α : Type} [Inhabited α] (M : Algebra α)
    (X O : RegionName) (flag : Bool) (epsilon : ℝ) (s : State α)
    (bound : s.regs .real [] "scale" = some (fun _ => M.literal (some .fp32) .real epsilon)) :
    exec M (view X O flag epsilon) s = exec M (source X O flag) s := by
  cases flag <;>
    simp [view,source,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,Option.bind,ComputeDType.eraseDType,State.setReg,bound]

/-- A misspelled binding must not silently return the unchanged source. -/
example : True := by
  fail_if_success
    have : ComputeKernel := source_scalar_view (source "x" "o" Bool.false) at "missing" := (0 : ℝ)
  trivial

#axiomsClean execution
end
end VeriTile.Tests.SourceScalarView

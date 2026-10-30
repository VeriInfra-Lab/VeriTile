import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.FPNarrowScope
open VeriTile.Triton FP.Structural
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def source (X : RegionName) : ComputeKernel := triton {
  tl.region X = tl.float16
  i = tl.arange(0, 2)
  x = tl.load(X + i)
  result = (tl.max(x, axis=0) / 127.0).to(tl.float16)
}

/-- A half destination does not erase the fp32 reduction/division inside it.
Both reduction and division stay in their own source compute context. -/
theorem narrow_keeps_fp32_child {α : Type} [Inhabited α] (M : Algebra α)
    (X : RegionName) (s : State α) :
    ∃ t, FP.Structural.exec M (source X) s = some t ∧
      t.regs .fp16 [] "result" = some (fun _ =>
        M.cast none .real .fp16
          (M.binary (some .fp32) .real .div
            (M.reduceMax (some .fp32) (shape := [2]) ⟨0, by simp⟩ Bool.false
              (fun i => M.cast (some .fp32) .fp16 .real ((s.mem X i.1.val).read .fp16)) PUnit.unit)
            (M.literal (some .fp32) .real 127))) := by
  simp [source, FP.Structural.exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, State.setReg, ofFloat, toFloat,
    Region.cast, ComputeDType.eraseDType, TileShape.axisDim, bop, numeric, Option.bind,
    show (127.0 : ℝ) = 127 by norm_num]

#axiomsClean narrow_keeps_fp32_child
end VeriTile.Tests.FPNarrowScope

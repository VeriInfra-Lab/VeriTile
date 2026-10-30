import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Semantics.Step
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.ReverseRange
open VeriTile.Triton
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false
set_option linter.unusedTactic false

/-- The source shape used by both reverse-cumsum kernels. -/
def chunks (T BT : Nat) : ComputeKernel := triton {
  for i in range(tl.cdiv($(T), $(BT)) - $(1), -$(1), -$(1)) {
    marker = $((7 : Int))
  }
}

noncomputable def realExec (k : ComputeKernel) (s : BlockState) : Option BlockState :=
  k.toAlgorithm?.toOption.bind (fun a => VeriTile.Triton.exec a s)

/-- Zero chunks must not execute even one body statement. -/
theorem empty_source (s : BlockState) : realExec (chunks 0 2) s = some s := by
  simp only [realExec, chunks, ComputeKernel.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?, Except.bind, Except.toOption,
    pure, bind, Option.bind_some, VeriTile.Triton.exec, stepStmts, stepStmt,
    VeriTile.Triton.evalOp, Tile.bop, Tile.scalar, NumericDType.sub,
    Broadcast.leftIndex, Broadcast.rightIndex]
  norm_num [stepForRangeIntAux]

theorem fp_empty_source {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    FP.Structural.exec M (chunks 0 2) s = some s := by
  simp only [chunks, FP.Structural.exec, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalOp, bind, Option.bind_some]
  simp only [pure, FP.Structural.numeric, FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Option.bind_some]
  norm_num only
  rw [FP.Structural.rangeInt]
  simp

/-- A stop of -2 and stride -2 produce 3, 1, -1, including the negative index. -/
def belowZero : ComputeKernel := triton {
  total = $((0 : Int))
  for i in range($(3), -$(2), -$(2)) {
    total = total + i
  }
}

/-- All three range operands are captured before the first iteration. -/
def changingBounds : ComputeKernel := triton {
  lo = $(4)
  hi = $(1)
  stride = $(2)
  total = $((0 : Int))
  for i in range(lo, -hi, -stride) {
    total = total + i
    lo = $(100)
    hi = $(100)
    stride = $(0)
  }
}

theorem real_respects_stop (s : BlockState) :
    (realExec belowZero s).map (fun t => t.regs .int [] "total") =
      some (some (Tile.scalar 3)) := by
  simp only [realExec, belowZero, ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    Except.bind, Except.toOption, pure, bind, Option.bind_some, VeriTile.Triton.exec]
  iterate 12
    all_goals try simp only [stepStmts, stepStmt, VeriTile.Triton.evalOp]
    all_goals try simp [BlockState.setReg, Tile.bop, Tile.scalar, NumericDType.add,
      NumericDType.sub, Broadcast.leftIndex, Broadcast.rightIndex, -Tile.scalar_eta]
    all_goals try rw [stepForRangeIntAux]

theorem fp_captures_bounds {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    (FP.Structural.exec M changingBounds s).map (fun t => t.regs .int [] "total") =
      some (some (fun _ => (6 : Int))) := by
  simp only [changingBounds, FP.Structural.exec]
  iterate 12
    all_goals try simp only [FP.Structural.run, FP.Structural.step,
      FP.Structural.evalExpr, FP.Structural.evalOp]
    all_goals try simp [FP.Structural.State.setReg, FP.Structural.numeric,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex]
    all_goals try rw [FP.Structural.rangeInt]
  rfl

#axiomsClean empty_source
#axiomsClean fp_empty_source
#axiomsClean real_respects_stop
#axiomsClean fp_captures_bounds
end VeriTile.Tests.ReverseRange

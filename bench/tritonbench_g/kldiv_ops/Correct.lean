import bench.tritonbench_g.kldiv_ops.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.KldivOps.Kernel.Correct
open VeriTile Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section

/-- The raw-target branch alone clamps to the runtime epsilon. -/
def loss (flag : Bool) (epsilon prediction target : ℝ) : ℝ :=
  if flag then Real.exp target * (target-prediction) else target*(Real.log (max target epsilon)-prediction)

def forwardFormula (p : Parameters) (flag : Bool) (red : Nat) (epsilon : ℝ)
    (xs : Interface.LogicalForwardInputs ℝ p) (row : Fin p.rows) (col : Fin (Interface.outputColumns p red)) : ℝ :=
  if red=0 then loss flag epsilon (Interface.columnAt xs.prediction row col.val) (Interface.columnAt xs.target row col.val)
  else ∑ k : Fin p.cols, loss flag epsilon (xs.prediction row k) (xs.target row k)

def backwardFormula (p : Parameters) (flag : Bool) (xs : Fin p.rows → Fin p.cols → ℝ) (row : Fin p.rows) (col : Fin p.cols) : ℝ :=
  if flag then -Real.exp (xs row col) else -(xs row col)

/-- Full forward row grid: pointwise output or one unnormalized row sum; both empty axes and read-only aliases remain. -/
specification forward (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ) (block : 0<p.block)
    (predictionSeparate : p.Y≠p.O) (targetSeparate : p.T≠p.O) :
    Spec.Real (Interface.logicalForwardIO RealExecution.algebra p flag red eps ⊨ forwardFormula p flag red eps) :=
  (LogicalReader.forward_full_iff p flag red eps).mpr (LogicalReader.forward p flag red eps block predictionSeparate targetSeparate)

/-- Complete unscaled prediction derivative, source safety and exact exterior frame. -/
specification backward (p : Parameters) (flag : Bool) (block : 0<p.block) (targetSeparate : p.T≠p.O) :
    Spec.Real (Interface.logicalBackwardIO RealExecution.algebra p flag ⊨ backwardFormula p flag) :=
  (LogicalReader.backward_full_iff p flag).mpr (LogicalReader.backward p flag block targetSeparate)
#axiomsClean forward
#axiomsClean backward

end
end VeriTile.Bench.Repairs.KldivOps.Kernel.Correct

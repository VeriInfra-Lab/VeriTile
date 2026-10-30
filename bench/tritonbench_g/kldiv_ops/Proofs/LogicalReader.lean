import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.kldiv_ops.IO
import bench.tritonbench_g.kldiv_ops.Proofs.IOReal

namespace VeriTile.Bench.Repairs.KldivOps.Kernel.LogicalReader
open VeriTile Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- Independent KL pointwise formula. Only the raw-target branch clamps to epsilon. -/
def loss (flag : Bool) (epsilon prediction target : ℝ) : ℝ :=
  if flag then Real.exp target * (target-prediction)
  else target * (Real.log (max target epsilon)-prediction)

/-- Elementwise output, or one unnormalized mathematical sum per row. -/
def forwardFormula (p : Parameters) (flag : Bool) (red : Nat) (epsilon : ℝ)
    (values : Interface.ForwardValues ℝ p) (i : Fin (Interface.outputLength p red)) : ℝ :=
  if red=0 then loss flag epsilon (Interface.predictionAt values i.val) (Interface.targetAt values i.val)
  else ∑ col : Fin p.cols, loss flag epsilon
    (Interface.predictionAt values (i.val*p.cols+col.val))
    (Interface.targetAt values (i.val*p.cols+col.val))

/-- Standalone backward kernel's unscaled prediction derivative. -/
def backwardFormula (p : Parameters) (flag : Bool) (values : Interface.BackwardValues ℝ p)
    (i : Fin (p.rows*p.cols)) : ℝ :=
  if flag then -Real.exp (Interface.backwardTargetAt values i.val)
  else -(Interface.backwardTargetAt values i.val)

/-- IO.lean exposes Y/T input spans, the full selected output span, every row,
the runtime epsilon binding, and the complete canonical forward source. -/
def forwardIO (p : Parameters) (flag : Bool) (red : Nat) (epsilon : ℝ) : ProgramIO ℝ ℝ :=
  Interface.forwardIO RealExecution.algebra p flag red epsilon id id

/-- Backward reads only T and writes every destination element. -/
def backwardIO (p : Parameters) (flag : Bool) : ProgramIO ℝ ℝ :=
  Interface.backwardIO RealExecution.algebra p flag id id

/-- Complete source access safety, mathematical outputs and exterior frame.
Read-only Y/T aliasing and both empty axes are retained. -/
theorem forward (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ)
    (block : 0<p.block) (predictionSeparate : p.Y≠p.O) (targetSeparate : p.T≠p.O) :
    Spec.Real (forwardIO p flag red eps ⊨ fun values =>
      Interface.result (forwardFormula p flag red eps values)) := by
  exact Ideal.forward p flag red eps block predictionSeparate targetSeparate

theorem backward (p : Parameters) (flag : Bool) (block : 0<p.block)
    (targetSeparate : p.T≠p.O) :
    Spec.Real (backwardIO p flag ⊨ fun values => Interface.result (backwardFormula p flag values)) := by
  exact Ideal.backward p flag block targetSeparate

end
end VeriTile.Bench.Repairs.KldivOps.Kernel.LogicalReader

namespace VeriTile.Bench.Repairs.KldivOps.Kernel.LogicalReader
open VeriTile Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section

def logicalForwardFormula (p : Parameters) (flag : Bool) (red : Nat) (epsilon : ℝ)
    (xs : Interface.LogicalForwardInputs ℝ p) (row : Fin p.rows) (col : Fin (Interface.outputColumns p red)) : ℝ :=
  if red=0 then loss flag epsilon (Interface.columnAt xs.prediction row col.val) (Interface.columnAt xs.target row col.val)
  else ∑ k : Fin p.cols, loss flag epsilon (xs.prediction row k) (xs.target row k)

def logicalBackwardFormula (p : Parameters) (flag : Bool) (xs : Fin p.rows → Fin p.cols → ℝ) (row : Fin p.rows) (col : Fin p.cols) : ℝ :=
  if flag then -Real.exp (xs row col) else -(xs row col)

theorem forward_full_iff (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ) :
    (Interface.logicalForwardIO RealExecution.algebra p flag red eps ⊨ logicalForwardFormula p flag red eps) ↔
      (forwardIO p flag red eps).Implements (fun xs => Interface.result (forwardFormula p flag red eps xs)) := by
  have target : (Interface.logicalForwardIO RealExecution.algebra p flag red eps).lift (logicalForwardFormula p flag red eps) =
      (fun xs => Interface.result (forwardFormula p flag red eps xs)) := by
    funext raw
    apply (Interface.logicalForwardOutputs ℝ p red).injective
    change (Interface.logicalForwardOutputs ℝ p red) ((Interface.logicalForwardOutputs ℝ p red).symm
      (logicalForwardFormula p flag red eps ((Interface.logicalForwardInputs ℝ p) raw))) = _
    rw [Equiv.apply_symm_apply]
    funext row col
    change logicalForwardFormula p flag red eps ((Interface.logicalForwardInputs ℝ p) raw) row col =
      forwardFormula p flag red eps raw ((Interface.logicalOutputIndex p red).symm (row,col))
    by_cases hz : red=0
    · subst red
      have hc : col.val<p.cols := col.isLt
      have h : (finProdFinEquiv (row,col)).val<p.rows*p.cols := (finProdFinEquiv (row,col)).isLt
      simp only [logicalForwardFormula,if_pos rfl,Interface.columnAt,dif_pos hc]
      change loss flag eps (raw (0 : Fin 2) (finProdFinEquiv (row,col),())) (raw (1 : Fin 2) (finProdFinEquiv (row,col),())) =
        loss flag eps (Interface.predictionAt raw (finProdFinEquiv (row,col)).val) (Interface.targetAt raw (finProdFinEquiv (row,col)).val)
      simp only [Interface.predictionAt,Interface.targetAt,dif_pos h,Interface.prediction,Interface.target]
      rfl
    · have hc : col.val=0 := by have small:=col.isLt; simp [Interface.outputColumns,hz] at small; omega
      have index : ((Interface.logicalOutputIndex p red).symm (row,col)).val = row.val := by
        change col.val+(Interface.outputColumns p red)*row.val = row.val
        simp only [Interface.outputColumns,if_neg hz,hc,one_mul,zero_add]
      simp only [logicalForwardFormula,if_neg hz,forwardFormula,index]
      apply Finset.sum_congr rfl
      intro k _
      have h : row.val*p.cols+k.val<p.rows*p.cols := by
        have hb := (finProdFinEquiv (row,k)).isLt
        simpa only [finProdFinEquiv_apply_val,Nat.add_comm,Nat.mul_comm] using hb
      simp only [Interface.predictionAt,Interface.targetAt,dif_pos h]
      have idx : finProdFinEquiv (row,k) = (⟨row.val*p.cols+k.val,h⟩ : Fin (p.rows*p.cols)) := by
        apply Fin.ext
        simp only [finProdFinEquiv_apply_val,Nat.add_comm,Nat.mul_comm]
      change loss flag eps (Interface.prediction raw (finProdFinEquiv (row,k))) (Interface.target raw (finProdFinEquiv (row,k))) = _
      rw [idx]
  change (forwardIO p flag red eps).Implements _ ↔ _
  rw [target]
  rfl

theorem backward_full_iff (p : Parameters) (flag : Bool) :
    (Interface.logicalBackwardIO RealExecution.algebra p flag ⊨ logicalBackwardFormula p flag) ↔
      (backwardIO p flag).Implements (fun xs => Interface.result (backwardFormula p flag xs)) := by
  have target : (Interface.logicalBackwardIO RealExecution.algebra p flag).lift (logicalBackwardFormula p flag) =
      (fun xs => Interface.result (backwardFormula p flag xs)) := by
    funext raw
    apply (Interface.logicalBackwardOutputs ℝ p).injective
    change (Interface.logicalBackwardOutputs ℝ p) ((Interface.logicalBackwardOutputs ℝ p).symm
      (logicalBackwardFormula p flag ((Interface.logicalBackwardInputs ℝ p) raw))) = _
    rw [Equiv.apply_symm_apply]
    funext row col
    change (if flag then -Real.exp (((Interface.logicalBackwardInputs ℝ p) raw) row col) else -(((Interface.logicalBackwardInputs ℝ p) raw) row col)) =
      backwardFormula p flag raw (finProdFinEquiv (row,col))
    simp only [backwardFormula,Interface.backwardTargetAt,dif_pos (finProdFinEquiv (row,col)).isLt,Interface.backwardTarget]
    rfl
  change (backwardIO p flag).Implements _ ↔ _
  rw [target]
  rfl

end
end VeriTile.Bench.Repairs.KldivOps.Kernel.LogicalReader

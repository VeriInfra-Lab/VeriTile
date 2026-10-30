import bench.tritonbench_g.kldiv_ops.Proofs.IOInterface
import bench.tritonbench_g.kldiv_ops.Proofs.RealFormula

namespace VeriTile.Bench.Repairs.KldivOps.Ideal
open VeriTile Triton FP.Structural Interface
noncomputable section

def forwardFormula (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ)
    (values : ForwardValues ℝ p) (i : Fin (outputLength p red)) : ℝ :=
  if red=0 then RealExecution.loss flag eps (predictionAt values i.val) (targetAt values i.val)
  else RealExecution.rowLoss flag eps (fun col => predictionAt values (i.val*p.cols+col))
    (fun col => targetAt values (i.val*p.cols+col)) p.cols

def backwardFormula (p : Parameters) (flag : Bool) (values : BackwardValues ℝ p)
    (i : Fin (p.rows*p.cols)) : ℝ :=
  if flag then -Real.exp (backwardTargetAt values i.val) else -(backwardTargetAt values i.val)

theorem forward (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ)
    (positive : 0<p.block) (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    (forwardIO RealExecution.algebra p flag red eps id id).Implements
      (fun values => result (forwardFormula p flag red eps values)) := by
  apply forward_implements _ p flag red eps positive id id
  intro values initial predictionRead targetRead
  obtain ⟨final,ran,outputs,frame⟩ := KernelResults.forward RealExecution.algebra p flag red eps initial positive hy ht
  refine ⟨final,ran,?_,frame⟩
  intro i
  rw [outputs i.val i.isLt]
  simp only [mapCell,mapValue,id_eq]
  congr 1
  by_cases mode : red=0
  · have hi : i.val<p.rows*p.cols := by simpa only [outputLength,if_pos mode] using i.isLt
    simp only [forwardFormula,if_pos mode,predictionRead _ hi,targetRead _ hi,id_eq]
    exact RealExecution.scalar_real flag eps _ _
  · simp only [forwardFormula,if_neg mode]
    rw [RealExecution.row_ideal flag eps p.cols p.block positive]
    apply Finset.sum_congr rfl
    intro col _
    have hr : i.val<p.rows := by simpa only [outputLength,if_neg mode] using i.isLt
    have hi := KernelResults.offset_lt p i.val col.val hr col.isLt
    dsimp only
    rw [predictionRead _ hi,targetRead _ hi]
    rfl

theorem backward (p : Parameters) (flag : Bool) (positive : 0<p.block) (ht : p.T≠p.O) :
    (backwardIO RealExecution.algebra p flag id id).Implements
      (fun values => result (backwardFormula p flag values)) := by
  apply backward_implements _ p flag positive id id
  intro values initial targetRead
  obtain ⟨final,ran,outputs,frame⟩ := KernelResults.backward RealExecution.algebra p flag initial positive ht
  refine ⟨final,ran,?_,frame⟩
  intro i
  rw [outputs i.val i.isLt,targetRead _ i.isLt]
  simp only [mapCell,mapValue,id_eq]
  congr 1
  exact RealExecution.backward_scalar_ideal flag _
end
end VeriTile.Bench.Repairs.KldivOps.Ideal

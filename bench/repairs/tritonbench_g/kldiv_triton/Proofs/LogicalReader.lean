import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.kldiv_triton.IO
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.PublicIO
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Real

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- The raw-target branch uses the probability-zero extension. -/
def loss (flag : Bool) (prediction target : ℝ) : ℝ :=
  if flag then Real.exp target * (target-prediction)
  else if target=0 then 0 else target*(Real.log target-prediction)

/-- Independent point losses or per-chunk sums; no host post-reduction. -/
def target (N B : Nat) (flag : Bool) (reduction : Nat)
    (prediction target : Nat → ℝ) (c : Nat) : ℝ :=
  if reduction=0 then loss flag (prediction c) (target c)
  else ∑ lane : Fin B, if c*B+lane.val<N then
    loss flag (prediction (c*B+lane.val)) (target (c*B+lane.val)) else 0


/-- Both complete FP32 source layouts are explicit in the shared IO data. -/
def forwardIO (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (work : List Nat) :=
  IO.forward RealModel.model rows sy st so N B flag reduction work
def backwardIO (rows so st N B : Nat) (flag : Bool) (work : List Nat) :=
  IO.backward RealModel.model rows so st N B flag work

/-- Complete forward grid: independent logical losses, actual source access
safety, successful execution and exact exterior frame for every full row order. -/
theorem forward (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=outputCount N B reduction) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    Spec.Real (forwardIO rows sy st so N B flag reduction work ⊨
      IO.forwardFormula (target N B flag reduction)) := by
  have target_eq (ys ts : Nat → ℝ) (c : Nat) :
      target N B flag reduction ys ts c = if reduction=0 then Math.loss flag (ys c) (ts c)
        else VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss flag ys ts N B c := by
    simp only [target, VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss,
      VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.paddedLoss_eq]
    split
    · rfl
    · apply Finset.sum_congr rfl
      intro lane _
      split
      · exact Math.loss_original flag _ _
      · rfl
  apply PublicIO.forward RealModel.model rows sy st so N B flag reduction positive unique work order
  intro ys ts c
  rw [target_eq]
  unfold KernelInputs.forwardValue
  split
  · exact RealModel.loss_real flag _ _
  · exact RealModel.chunk_real flag N B c ys ts

/-- Complete backward grid computes the unscaled derivative. -/
theorem backward (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (IO.rowAddress (rows:=rows) (columns:=N) so))
    (work : List Nat) (order : work.Perm (List.range rows)) :
    Spec.Real (backwardIO rows so st N B flag work ⊨ IO.backwardFormula (fun t => if flag then -Real.exp t else -t)) := by
  apply PublicIO.backward RealModel.model rows so st N B flag positive unique work order
  exact RealModel.gradient_real flag

end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.LogicalReader

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
open scoped LogicalProgramIO
noncomputable section

def logicalForwardFormula (rows N B : Nat) (flag : Bool) (reduction : Nat)
    (xs : IO.LogicalForwardInputs ℝ rows N) (row : Fin rows) (col : Fin (outputCount N B reduction)) : ℝ :=
  target N B flag reduction (IO.extendRow xs.prediction row) (IO.extendRow xs.target row) col.val

theorem forward_full_iff (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (work : List Nat) :
    (IO.logicalForwardIO RealModel.model rows sy st so N B flag reduction work ⊨ logicalForwardFormula rows N B flag reduction) ↔
      (forwardIO rows sy st so N B flag reduction work).Implements (IO.forwardFormula (target N B flag reduction)) := by
  have eq : (IO.logicalForwardIO RealModel.model rows sy st so N B flag reduction work).lift
      (logicalForwardFormula rows N B flag reduction) = IO.forwardFormula (target N B flag reduction) := by
    funext xs o; fin_cases o; funext j
    change target N B flag reduction
      (IO.extendRow (fun r c => xs ⟨0,by change 0<2; omega⟩ (r,c,PUnit.unit)) j.1)
      (IO.extendRow (fun r c => xs ⟨1,by change 1<2; omega⟩ (r,c,PUnit.unit)) j.1) j.2.1.val =
      target N B flag reduction (IO.rowsOf (xs ⟨0,by change 0<2; omega⟩) j.1.val)
        (IO.rowsOf (xs ⟨1,by change 1<2; omega⟩) j.1.val) j.2.1.val
    congr 1 <;> funext c <;> simp only [IO.extendRow,IO.rowsOf,dif_pos j.1.isLt]
  change (forwardIO rows sy st so N B flag reduction work).Implements _ ↔ _
  simpa only [eq]

theorem backward_full_iff (rows so st N B : Nat) (flag : Bool) (work : List Nat) :
    (IO.logicalBackwardIO RealModel.model rows so st N B flag work ⊨ fun xs row col =>
      if flag then -Real.exp (xs row col) else -(xs row col)) ↔
      (backwardIO rows so st N B flag work).Implements (IO.backwardFormula (fun t => if flag then -Real.exp t else -t)) := by
  have eq : (IO.logicalBackwardIO RealModel.model rows so st N B flag work).lift
      (fun xs row col => if flag then -Real.exp (xs row col) else -(xs row col)) =
        IO.backwardFormula (fun t => if flag then -Real.exp t else -t) := by
    funext xs o; fin_cases o; funext j; rfl
  change (backwardIO rows so st N B flag work).Implements _ ↔ _
  simpa only [eq]

end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.LogicalReader

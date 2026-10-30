import bench.repairs.tritonbench_g.kldiv_triton.Proofs.KernelInterface
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.Real

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.RawCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
noncomputable section

/-- Independent point losses or per-chunk sums; no host post-reduction. -/
def target (N B : Nat) (flag : Bool) (reduction : Nat)
    (prediction target : Nat → ℝ) (c : Nat) : ℝ :=
  if reduction=0 then Math.loss flag (prediction c) (target c)
  else VeriTile.Bench.TritonBenchG.KldivTriton.FullMath.chunkLoss flag prediction target N B c

/-- Complete raw forward grid, including every per-chunk/element output and
all exterior typed cells, for every complete row schedule. -/
theorem forward (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows) (cols:=outputCount N B reduction) so))
    (before : State ℝ) (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,RawGrid.forwardRun (RealModel.model) sy st so N B flag reduction schedule before=some final ∧
      GridExecution.Post "loss_ptr" (RawGrid.address (rows:=rows) (cols:=outputCount N B reduction) so)
        (fun x => .mk .real (target N B flag reduction
          (Tile.input before "y_ptr" (x.1.val*sy)) (Tile.input before "gt_ptr" (x.1.val*st)) x.2.val))
        (fun _ => True) before final := by
  obtain ⟨final,ran,values,frame⟩ := RawGrid.forward (RealModel.model) rows sy st so N B flag reduction positive unique before schedule order
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  apply congrArg (fun v : ℝ => (Cell.mk .real v : Cell ℝ))
  unfold RawGrid.forwardValue target
  split
  · exact RealModel.loss_real flag _ _
  · exact RealModel.chunk_real flag N B _ _ _

/-- Complete raw backward grid with the unscaled derivative and full frame. -/
theorem backward (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows) (cols:=N) so))
    (before : State ℝ) (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,RawGrid.backwardRun (RealModel.model) so st N B flag schedule before=some final ∧
      GridExecution.Post "input_ptr" (RawGrid.address (rows:=rows) (cols:=N) so)
        (fun x => .mk .real (Math.gradient flag (Tile.input before "target_ptr" (x.1.val*st) x.2.val)))
        (fun _ => True) before final := by
  obtain ⟨final,ran,values,frame⟩ := RawGrid.backward (RealModel.model) rows so st N B flag positive unique before schedule order
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  apply congrArg (fun v : ℝ => (Cell.mk .real v : Cell ℝ))
  exact RealModel.gradient_real flag _
#specNonCircular target avoiding [forwardSource,backwardSource,RawGrid.forwardRun,RawGrid.backwardRun]
#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.RawCorrect

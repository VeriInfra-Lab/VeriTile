import bench.repairs.tritonbench_g.kldiv_triton.Proofs.KernelInterface
import bench.repairs.tritonbench_g.kldiv_triton.Proofs.SourceNative

namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.RawNative
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
noncomputable section

/-- Independent point losses or per-chunk sums; no host post-reduction. -/
def target (p : SourceNative.Primitives) (plans : SourceNative.Plans) (N B : Nat) (flag : Bool) (reduction : Nat)
    (prediction target : Nat → SourceNative.Word) (c : Nat) : SourceNative.Word :=
  if reduction=0 then SourceNative.loss p flag (prediction c) (target c)
  else SourceNative.chunk p plans flag N B c prediction target

/-- Complete raw forward grid, including every per-chunk/element output and
all exterior typed cells, for every complete row schedule. -/
theorem forward (p : SourceNative.Primitives) (plans : SourceNative.Plans) (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat)
    (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows) (cols:=outputCount N B reduction) so))
    (before : State SourceNative.Word) (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,RawGrid.forwardRun (SourceNative.model p plans) sy st so N B flag reduction schedule before=some final ∧
      GridExecution.Post "loss_ptr" (RawGrid.address (rows:=rows) (cols:=outputCount N B reduction) so)
        (fun x => .mk .real (target p plans N B flag reduction
          (Tile.input before "y_ptr" (x.1.val*sy)) (Tile.input before "gt_ptr" (x.1.val*st)) x.2.val))
        (fun _ => True) before final := by
  obtain ⟨final,ran,values,frame⟩ := RawGrid.forward (SourceNative.model p plans) rows sy st so N B flag reduction positive unique before schedule order
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  apply congrArg (fun v : SourceNative.Word => (Cell.mk .real v : Cell SourceNative.Word))
  unfold RawGrid.forwardValue target
  split
  · exact SourceNative.loss_lowering p plans flag _ _
  · exact SourceNative.chunk_lowering p plans flag N B _ _ _

/-- Complete raw backward grid with the unscaled derivative and full frame. -/
theorem backward (p : SourceNative.Primitives) (plans : SourceNative.Plans) (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows) (cols:=N) so))
    (before : State SourceNative.Word) (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,RawGrid.backwardRun (SourceNative.model p plans) so st N B flag schedule before=some final ∧
      GridExecution.Post "input_ptr" (RawGrid.address (rows:=rows) (cols:=N) so)
        (fun x => .mk .real (SourceNative.derivative p flag (Tile.input before "target_ptr" (x.1.val*st) x.2.val)))
        (fun _ => True) before final := by
  obtain ⟨final,ran,values,frame⟩ := RawGrid.backward (SourceNative.model p plans) rows so st N B flag positive unique before schedule order
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  apply congrArg (fun v : SourceNative.Word => (Cell.mk .real v : Cell SourceNative.Word))
  exact SourceNative.derivative_lowering p plans flag _
#specNonCircular target avoiding [forwardSource,backwardSource,RawGrid.forwardRun,RawGrid.backwardRun]
#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.RawNative

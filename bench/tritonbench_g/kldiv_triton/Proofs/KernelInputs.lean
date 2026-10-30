import bench.tritonbench_g.kldiv_triton.Proofs.KernelInterface

/-! Logical input binding for the complete raw grid. These lemmas reuse the
existing source execution and frame proofs. Only active input coordinates are
pinned; padded reduction lanes introduce no input assumptions. -/
namespace VeriTile.Bench.Repairs.KldivTriton.Kernel.KernelInputs
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.IntScaledMatmul.Revision001
open VeriTile.Bench.TritonBenchG.KldivTriton.FullMath (outputCount)
noncomputable section

def forwardValue {α : Type} (m : Tile.Model α) (N B : Nat) (flag : Bool)
    (reduction : Nat) (prediction target : Nat → α) (c : Nat) : α :=
  if reduction=0 then Tile.loss m flag (m.alg.fp32Load (prediction c))
    (m.alg.fp32Load (target c))
  else Tile.chunkSum m flag N B c prediction target

theorem forward {α : Type} [Inhabited α] (m : Tile.Model α)
    (rows sy st so N B : Nat) (flag : Bool) (reduction : Nat) (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows)
      (cols:=outputCount N B reduction) so))
    (prediction target : Nat → Nat → α) (before : State α)
    (predictionReads : ∀ r, r<rows → ∀ c, c<N →
      Tile.input before "y_ptr" (r*sy) c=prediction r c)
    (targetReads : ∀ r, r<rows → ∀ c, c<N →
      Tile.input before "gt_ptr" (r*st) c=target r c)
    (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,RawGrid.forwardRun m sy st so N B flag reduction schedule before=some final ∧
      GridExecution.Post "loss_ptr"
        (RawGrid.address (rows:=rows) (cols:=outputCount N B reduction) so)
        (fun x => .mk .real (forwardValue m N B flag reduction
          (prediction x.1.val) (target x.1.val) x.2.val))
        (fun _ => True) before final := by
  obtain ⟨final,ran,values,frame⟩ :=
    RawGrid.forward m rows sy st so N B flag reduction positive unique before schedule order
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  apply congrArg (fun v : α => (Cell.mk .real v : Cell α))
  unfold RawGrid.forwardValue forwardValue
  by_cases reduced : reduction=0
  · have active : x.2.val<N := by simpa [outputCount,reduced] using x.2.isLt
    simp only [reduced,if_pos, predictionReads _ x.1.isLt _ active,
      targetReads _ x.1.isLt _ active]
  · simp only [reduced,if_false]
    exact Tile.chunkSum_congr m flag N B x.2.val _ _ _ _
      (predictionReads _ x.1.isLt) (targetReads _ x.1.isLt)

theorem backward {α : Type} [Inhabited α] (m : Tile.Model α)
    (rows so st N B : Nat) (flag : Bool) (positive : 0<B)
    (unique : Function.Injective (RawGrid.address (rows:=rows) (cols:=N) so))
    (target : Nat → Nat → α) (before : State α)
    (targetReads : ∀ r, r<rows → ∀ c, c<N →
      Tile.input before "target_ptr" (r*st) c=target r c)
    (schedule : List Nat) (order : schedule.Perm (List.range rows)) :
    ∃ final,RawGrid.backwardRun m so st N B flag schedule before=some final ∧
      GridExecution.Post "input_ptr" (RawGrid.address (rows:=rows) (cols:=N) so)
        (fun x => .mk .real (Tile.gradient m flag (m.alg.fp32Load (target x.1.val x.2.val))))
        (fun _ => True) before final := by
  obtain ⟨final,ran,values,frame⟩ :=
    RawGrid.backward m rows so st N B flag positive unique before schedule order
  refine ⟨final,ran,?_,frame⟩
  intro x _
  rw [values x trivial]
  simp only [RawGrid.backwardValue,targetReads _ x.1.isLt _ x.2.isLt]

#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.KldivTriton.Kernel.KernelInputs

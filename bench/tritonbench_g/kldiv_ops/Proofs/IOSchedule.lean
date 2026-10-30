import bench.tritonbench_g.kldiv_ops.IO
import bench.tritonbench_g.kldiv_ops.Proofs.SafetyForward
import bench.tritonbench_g.kldiv_ops.Proofs.KernelResults

namespace VeriTile.Bench.Repairs.KldivOps.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
variable {α β : Type} [Inhabited α] [Inhabited β]

@[simp] theorem forward_initial (M : Algebra β) (p : Parameters) (flag : Bool) (red : Nat)
    (eps : β) (row : Nat) (s : State β) :
    (forwardInvocation M p flag red eps row).initial s=Tile.bindEpsilon eps (Program.atRow s row) := rfl
@[simp] theorem backward_initial (M : Algebra β) (p : Parameters) (flag : Bool)
    (row : Nat) (s : State β) :
    (backwardInvocation M p flag row).initial s=Program.atRow s row := rfl

theorem forward_schedule_run (M : Algebra β) (p : Parameters) (flag : Bool) (red : Nat)
    (eps : β) (rows : List Nat) (s : State β) :
    runSchedule (rows.map (forwardInvocation M p flag red eps)) s=
      VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.runSchedule
        (fun row current => run (engine M) (forward p flag red).surfaceBody
          (Tile.bindEpsilon eps (Program.atRow current row))) rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,forwardInvocation,KernelDSL.forward_eq,
      VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.runSchedule]
    change (run (engine M) (forward p flag red).surfaceBody
      (Tile.bindEpsilon eps (Program.atRow s row))).bind _ = _
    congr 1
    funext t
    exact ih t

theorem backward_schedule_run (M : Algebra β) (p : Parameters) (flag : Bool)
    (rows : List Nat) (s : State β) :
    runSchedule (rows.map (backwardInvocation M p flag)) s=
      VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.runSchedule
        (fun row current => run (engine M) (backward p flag).surfaceBody
          (Program.atRow current row)) rows s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,backwardInvocation,KernelDSL.backward_eq,
      VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.runSchedule]
    change (run (engine M) (backward p flag).surfaceBody (Program.atRow s row)).bind _ = _
    congr 1
    funext t
    exact ih t

theorem forward_schedule_safe (M : Algebra β) (p : Parameters) (flag : Bool) (red : Nat)
    (eps : β) (positive : 0<p.block) (rows : List Nat) (members : ∀ row∈rows,row<p.rows)
    (bounds : RegionName → Nat)
    (hy : ∀ i,i<p.rows*p.cols → i<bounds p.Y)
    (ht : ∀ i,i<p.rows*p.cols → i<bounds p.T)
    (ho : ∀ i,i<outputLength p red → i<bounds p.O) (s : State β) :
    ScheduleSafe bounds (rows.map (forwardInvocation M p flag red eps)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨?_,fun t _ => ih (fun r hr => members r (by simp [hr])) t⟩
    change FP.Structural.TraceSafe (engine M) bounds (KernelDSL.forward p flag red)
      (Tile.bindEpsilon eps (Program.atRow s row))
    apply SafetyForward.source_safe M p flag red eps (Program.atRow s row) positive bounds
    · intro col active
      exact hy _ (KernelResults.offset_lt p row col (members row (by simp)) active)
    · intro col active
      exact ht _ (KernelResults.offset_lt p row col (members row (by simp)) active)
    · intro mode col active
      apply ho
      simpa only [outputLength,mode,ite_true] using
        KernelResults.offset_lt p row col (members row (by simp)) active
    · intro mode
      apply ho
      simpa only [outputLength,mode,ite_false] using members row (by simp)

theorem backward_schedule_safe (M : Algebra β) (p : Parameters) (flag : Bool)
    (positive : 0<p.block) (rows : List Nat) (members : ∀ row∈rows,row<p.rows)
    (bounds : RegionName → Nat)
    (ht : ∀ i,i<p.rows*p.cols → i<bounds p.T)
    (ho : ∀ i,i<p.rows*p.cols → i<bounds p.O) (s : State β) :
    ScheduleSafe bounds (rows.map (backwardInvocation M p flag)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨?_,fun t _ => ih (fun r hr => members r (by simp [hr])) t⟩
    change FP.Structural.TraceSafe (engine M) bounds (KernelDSL.backward p flag)
      (Program.atRow s row)
    apply SafetyBackward.source_safe M p flag (Program.atRow s row) positive bounds
    · intro col active
      exact ht _ (KernelResults.offset_lt p row col (members row (by simp)) active)
    · intro col active
      exact ho _ (KernelResults.offset_lt p row col (members row (by simp)) active)

end
end VeriTile.Bench.Repairs.KldivOps.Interface

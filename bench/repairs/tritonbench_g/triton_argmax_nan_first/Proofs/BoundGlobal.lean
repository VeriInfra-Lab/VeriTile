import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.BoundSource

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.BoundSource
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.TritonArgmax Source Memory
noncomputable section

/-- Adjacent launches share exactly the same intermediate physical state. -/
inductive Steps {M α : Type} (step : α → M → M → Prop) : List α → M → M → Prop
  | nil (s) : Steps step [] s s
  | cons {job jobs s t u} : step job s t → Steps step jobs t u → Steps step (job::jobs) s u

theorem steps_fold {M α : Type} (step : α → M → M → Prop) (run : M → α → M)
    (jobs : List α) (s : M) (each : ∀ state job,job∈jobs → step job state (run state job)) :
    Steps step jobs s (jobs.foldl run s) := by
  induction jobs generalizing s with
  | nil => exact .nil s
  | cons job jobs ih =>
    exact .cons (each s job (by simp)) (ih _ (fun state j hj => each state j (by simp [hj])))

def GlobalRun {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (rest : Algebra Word) (template : State Word) (c : GlobalConfig) (p : GlobalMemory.Buffers)
    (index64 : Bool) (before after : M) : Prop :=
  ∃ middle, Steps (FirstStep mem binding rest template c p index64) (List.range c.blocks) before middle ∧
    SecondStep mem binding rest template c p middle after

theorem first_grid_source {M d} (mem : Primitives M) (scratch : Scratch d)
    (c : GlobalConfig) (g : Native.GlobalPrograms c) (p : GlobalMemory.Buffers)
    (positive : 0<c.block) (active : ∀ pid,pid<c.blocks → pid*c.block<c.size) (before : M) :
    (List.range c.blocks).foldl (Native.firstProgram mem scratch g p) before=
      GlobalMemory.firstGrid mem scratch.scalars c p before := by
  have one : ∀ state pid,pid<c.blocks → Native.firstProgram mem scratch g p state pid=
      GlobalMemory.firstLive mem scratch.scalars c p state pid := by
    intro state pid hp
    simp only [Native.firstProgram,Native.first_value scratch g _ pid positive (active pid hp),Native.first_index scratch g]
    rfl
  have fold : ∀ jobs : List Nat,(∀ pid,pid∈jobs → pid<c.blocks) → ∀ state,
      jobs.foldl (Native.firstProgram mem scratch g p) state=
        jobs.foldl (GlobalMemory.firstLive mem scratch.scalars c p) state := by
    intro jobs valid
    induction jobs with
    | nil => intro state; rfl
    | cons pid jobs ih =>
      intro state
      rw [List.foldl_cons,List.foldl_cons,one state pid (valid pid (by simp))]
      exact ih (fun j hj => valid j (by simp [hj])) _
  exact fold _ (fun pid hp => List.mem_range.mp hp) before

theorem global_run {M casts kind} (mem : Primitives M) (binding : Binding casts kind)
    (rest : Algebra Word) (template : State Word) (c : GlobalConfig) (g : Native.GlobalPrograms c)
    (p : GlobalMemory.Buffers) (layout : GlobalMemory.Layout kind.format c p)
    (positive : 0<c.block) (midPositive : 0<c.midBlock)
    (active : ∀ pid,pid<c.blocks → pid*c.block<c.size) (index64 : Bool) (before : M) :
    GlobalRun mem binding rest template c p index64 before
      (Native.global mem (Request.scratch casts kind) kind.isFloating g p before) := by
  let middle := (List.range c.blocks).foldl (Native.firstProgram mem (Request.scratch casts kind) g p) before
  refine ⟨middle,steps_fold _ _ _ before (fun state pid _ =>
    first_step mem binding rest template c g p positive index64 pid state),?_⟩
  apply second_step mem binding rest template c g p midPositive middle
  intro i hi
  dsimp only [middle]
  rw [first_grid_source mem (Request.scratch casts kind) c g p positive active,
    GlobalMemory.grid_values mem (Request.scratch casts kind).scalars c p layout before i hi]
  cases binding with
  | floating k round codec => exact TypedScalarBinding.stage1_reachable k (casts.floating k) c _ i
  | integer k zero => trivial

#axiomsClean global_run
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.BoundSource

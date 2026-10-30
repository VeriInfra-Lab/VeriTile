import bench.tritonbench_g.sgmv_expand_slice.Proofs.ScheduleIO
import bench.tritonbench_g.sgmv_expand_slice.Sources.Kernels
import bench.tritonbench_g.sgmv_expand_slice.Proofs.Geometry

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
open VeriTile.Triton

/-- Host batch order is preserved. The max-sequence hint is not used to
truncate coverage. Empty/sentinel/zero-column segments have no jobs. -/
noncomputable def runSourceJobs {α : Type} [Inhabited α]
    (M : FP.Structural.Algebra α) (abi : SourceABI) (p : KernelArgs) :
    List Job → FP.Structural.State α → Option (FP.Structural.State α)
  | [],s => some s
  | j::js,s => do
      let one := {s with
        pids := fun ax => if ax = 0 then j.pid else 0
        numPids := fun ax => if ax = 0 then j.count else 1}
      let next ← FP.Structural.exec M (kernel abi {p with batch := j.batch}) one
      runSourceJobs M abi p js next

/-- Real projection is explicit and fallible. Projection failure is not turned
into an empty successful program. No FP numerical theorem enters this runner. -/
noncomputable def runRealJobs (abi : SourceABI) (p : KernelArgs) :
    List Job → BlockState → Option BlockState
  | [],s => some s
  | j::js,s => do
      let projected ← match (kernel abi {p with batch := j.batch}).eraseDType.toAlgorithm? with
        | .ok k => some k | .error _ => none
      let one := {s with
        pids := fun ax => if ax = 0 then j.pid else 0
        numPids := fun ax => if ax = 0 then j.count else 1}
      let next ← VeriTile.Triton.exec projected one
      runRealJobs abi p js next

@[simp] theorem jobs_nil (columns : Nat) : jobs columns [] = [] := rfl
@[simp] theorem source_no_jobs {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (abi : SourceABI) (p : KernelArgs) (s : FP.Structural.State α) :
    runSourceJobs M abi p [] s = some s := rfl
@[simp] theorem real_no_jobs (abi : SourceABI) (p : KernelArgs) (s : BlockState) :
    runRealJobs abi p [] s = some s := rfl

theorem source_jobs_append {α : Type} [Inhabited α] (M : FP.Structural.Algebra α)
    (abi : SourceABI) (p : KernelArgs) (before after : List Job) (s : FP.Structural.State α) :
    runSourceJobs M abi p (before++after) s =
      (runSourceJobs M abi p before s).bind (runSourceJobs M abi p after) := by
  induction before generalizing s with
  | nil => rfl
  | cons j js ih =>
      simp only [List.cons_append,runSourceJobs]
      cases h : FP.Structural.exec M (kernel abi {p with batch := j.batch})
        {s with
          pids := fun ax => if ax = 0 then j.pid else 0
          numPids := fun ax => if ax = 0 then j.count else 1} <;> simp [h,ih]


end VeriTile.Bench.Repairs.SgmvExpandSlice

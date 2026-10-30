import bench.tritonbench_g.softmax_flaggems.Proofs.GridIO
import VeriTile.Triton.Float.ProgramIO

/-! Concrete whole-grid invocations of the unchanged source. The public schedule
installs exactly the existing grid metadata and clears only local registers. -/
namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural
open SoftmaxFlaggemsFiniteSeed
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

@[simp] theorem job_initial (A : Algebra α) (source : ComputeKernel) (gx gy : Nat)
    (p : Program) (s : State α) : (job A source gx gy p).initial s=FPGrid.setup gx gy p s := rfl

theorem run_jobs (A : Algebra α) (source : ComputeKernel) (gx gy : Nat)
    (ps : List Program) (s : State α) :
    runSchedule (ps.map (job A source gx gy)) s=FPGrid.run A source gx gy ps s := by
  induction ps generalizing s with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.map_cons,runSchedule,Invocation.run,FPGrid.run,job,ih]
    rfl

@[simp] theorem forwardJobs_run (A : Algebra α) (X Y : RegionName) (d : Dimensions)
    (c : LaunchConfig) (s : State α) :
    runSchedule (forwardJobs A X Y d c) s=FPGrid.forwardRun A X Y d c s :=
  run_jobs A _ _ _ _ s

@[simp] theorem backwardJobs_run (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (c : LaunchConfig) (s : State α) :
    runSchedule (backwardJobs A P G Y d c) s=FPGrid.backwardRun A P G Y d c s :=
  run_jobs A _ _ _ _ s

theorem jobs_safe (A : Algebra α) (source : ComputeKernel) (gx gy : Nat)
    (ps : List Program) (bounds : RegionName → Nat)
    (h : ∀ p∈ps, ∀ s,TraceSafe A bounds source (FPGrid.setup gx gy p s)) (s : State α) :
    ScheduleSafe bounds (ps.map (job A source gx gy)) s := by
  induction ps generalizing s with
  | nil => trivial
  | cons p ps ih =>
    refine ⟨h p (by simp) s,?_⟩
    intro t _
    exact ih (fun q hq => h q (by simp [hq])) t

end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface

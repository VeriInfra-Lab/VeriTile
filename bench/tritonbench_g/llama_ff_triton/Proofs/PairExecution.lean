import bench.tritonbench_g.llama_ff_triton.PairSchedule
import bench.tritonbench_g.llama_ff_triton.Proofs.PairSource
import bench.tritonbench_g.llama_ff_triton.Proofs.IOResults

namespace VeriTile.Bench.Repairs.LlamaFFScalarCast.PairExecution
open VeriTile.Triton FP.Structural Execution Interface
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
variable {α : Type} [Inhabited α]



/-- The full selected grid, with the existing transformed source in every CTA. -/

theorem original_schedule (a : Arithmetic α) (g : GateInstructions α) (p : Parameters)
    (xs : Inputs α) :
    (programIO Bool.false a g p).schedule xs =
      (List.range (Grid.count p)).map (originalJob a p) := by
  change (List.range (Grid.count p)).flatMap (fun pid => [originalJob a p pid]) = _
  exact List.map_eq_flatMap.symm

theorem job_run (a : Arithmetic α) (p : Parameters) (pid : Nat) (s : State α) :
    (candidateJob a p pid).run s = (originalJob a p pid).run s := by
  exact (PairSource.exact a.model p ((originalJob a p pid).initial s)).symm

theorem job_safe (a : Arithmetic α) (p : Parameters) (pid : Nat)
    (bounds : RegionName → Nat) (s : State α) :
    (originalJob a p pid).Safe bounds s ↔ (candidateJob a p pid).Safe bounds s := by
  exact PairSource.safe_iff a.model p ((originalJob a p pid).initial s) bounds

theorem schedule_exact (a : Arithmetic α) (p : Parameters) (pids : List Nat)
    (s : State α) :
    runSchedule (pids.map (candidateJob a p)) s =
      runSchedule (pids.map (originalJob a p)) s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons, runSchedule, job_run]
    cases h : (originalJob a p pid).run s <;> simp [h,ih]

theorem schedule_safe (a : Arithmetic α) (p : Parameters) (pids : List Nat)
    (bounds : RegionName → Nat) (s : State α) :
    ScheduleSafe bounds (pids.map (originalJob a p)) s ↔
      ScheduleSafe bounds (pids.map (candidateJob a p)) s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [List.map_cons, ScheduleSafe, job_safe, job_run, ih]

theorem complete_run (a : Arithmetic α) (g : GateInstructions α) (p : Parameters)
    (xs : Inputs α) (initial : State α) :
    runSchedule (candidateSchedule a p) initial =
      runSchedule ((programIO Bool.false a g p).schedule xs) initial := by
  rw [original_schedule]
  exact schedule_exact a p _ initial

theorem complete_safe (a : Arithmetic α) (g : GateInstructions α) (p : Parameters)
    (xs : Inputs α) (bounds : RegionName → Nat) (initial : State α) :
    ScheduleSafe bounds ((programIO Bool.false a g p).schedule xs) initial ↔
      ScheduleSafe bounds (candidateSchedule a p) initial := by
  rw [original_schedule]
  exact schedule_safe a p _ bounds initial

#axiomsClean original_schedule
#axiomsClean schedule_exact
#axiomsClean schedule_safe
#axiomsClean complete_run
#axiomsClean complete_safe
end
end VeriTile.Bench.Repairs.LlamaFFScalarCast.PairExecution

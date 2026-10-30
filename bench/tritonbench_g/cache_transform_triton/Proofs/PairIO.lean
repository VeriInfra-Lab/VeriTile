import bench.tritonbench_g.cache_transform_triton.Proofs.PairConnection

namespace VeriTile.Bench.Repairs.CacheTransform.PairIO
open VeriTile Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
variable {A : Type} [Inhabited A]

theorem job_run (M : Algebra A) (p : Parameters) (pid : Nat) (s : State A) :
    (originalJob M p pid).run s = (candidateJob M p pid).run s :=
  PairConnection.execution M p _

theorem job_safe (M : Algebra A) (p : Parameters) (pid : Nat) (bounds : RegionName → Nat) (s : State A) :
    (originalJob M p pid).Safe bounds s ↔ (candidateJob M p pid).Safe bounds s :=
  PairConnection.safety M p _ bounds

theorem schedule_run (M : Algebra A) (p : Parameters) (pids : List Nat) (s : State A) :
    runSchedule (pids.map (originalJob M p)) s = runSchedule (pids.map (candidateJob M p)) s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid pids ih =>
    simp only [List.map_cons,runSchedule,job_run]
    cases (candidateJob M p pid).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe (M : Algebra A) (p : Parameters) (pids : List Nat)
    (bounds : RegionName → Nat) (s : State A) :
    ScheduleSafe bounds (pids.map (originalJob M p)) s ↔
      ScheduleSafe bounds (pids.map (candidateJob M p)) s := by
  induction pids generalizing s with
  | nil => rfl
  | cons pid pids ih =>
    simp only [List.map_cons,ScheduleSafe,job_safe,job_run]
    exact and_congr_right (fun _ => forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Both full byte-gather grids execute safely and successfully. Every output
byte/tag and both complete exterior memory frames are preserved. Exact input
cells retain the original signed metadata and byte admission scope. -/
theorem equivalent (M : Algebra A) (p : Parameters) (n capacity : Nat)
    (start : p.start=0) (block : p.block=256) (grid : blocks p.count≤gridLimit) :
    (cacheIO M p n capacity).EquivalentOnCells (candidate M p n capacity)
      (CacheIO.Domain M p n capacity) := by
  intro xs domain bounds hb s hin
  obtain ⟨safe,final,ran,values,frame⟩ := IOProof.implements M p start block grid n capacity xs domain bounds hb s hin
  change ScheduleSafe bounds ((List.range (blocks p.count)).map (originalJob M p)) s at safe
  change runSchedule ((List.range (blocks p.count)).map (originalJob M p)) s=some final at ran
  refine ⟨safe,(schedule_safe M p _ bounds s).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_run M p _ s).symm.trans ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.CacheTransform.PairIO

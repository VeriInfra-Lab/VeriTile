import bench.repairs.tritonbench_g.rope_transform.PairIO
import bench.repairs.tritonbench_g.rope_transform.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.RopeTransform.PairIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
variable {α β : Type} [Inhabited α] [Inhabited β]

theorem invocation_run (M : Algebra β) (r : Request) (row : Nat) (s : State β) :
    (candidateInvocation M r row).run s = (PublicIO.invocation M r row).run s :=
  (IndexCSE.execution M r.parameters ((PublicIO.invocation M r row).initial s)).symm

theorem invocation_safe (M : Algebra β) (r : Request) (row : Nat)
    (bounds : RegionName → Nat) (s : State β) :
    (PublicIO.invocation M r row).Safe bounds s ↔
      (candidateInvocation M r row).Safe bounds s :=
  IndexCSE.trace_safe M r.parameters bounds ((PublicIO.invocation M r row).initial s)

theorem schedule_run (M : Algebra β) (r : Request) (rows : List Nat) (s : State β) :
    runSchedule (rows.map (candidateInvocation M r)) s =
      runSchedule (rows.map (PublicIO.invocation M r)) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    exact congrArg (fun f => (PublicIO.invocation M r row).run s >>= f) (funext ih)

theorem schedule_safe (M : Algebra β) (r : Request) (rows : List Nat)
    (bounds : RegionName → Nat) (s : State β) :
    ScheduleSafe bounds (rows.map (PublicIO.invocation M r)) s ↔
      ScheduleSafe bounds (rows.map (candidateInvocation M r)) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_run]
    exact and_congr (invocation_safe M r row bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Complete candidate grid: both Q/K outputs, every original payload, source
access safety, successful execution and the entire exterior raw-cell frame. -/
theorem candidate_implements (M : Algebra β) (encode : α → β)
    (observe : β → α) (r : Request) :
    (candidate M encode observe r).toProgramIO.Implements (PublicIO.scalarTarget M encode observe r) := by
  intro xs admitted bounds backed initial inputs
  obtain ⟨safe,final,ran,outputs,frame⟩ := PublicIO.implements M encode observe r xs admitted bounds backed initial inputs
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (schedule_safe M r (List.range r.rows) bounds initial).mp safe
  · exact (schedule_run M r (List.range r.rows) initial).trans ran

theorem equivalent (M : Algebra β) (encode : α → β) (observe : β → α) (r : Request) :
    (PublicIO.sourceIO M encode observe r).Equivalent (candidate M encode observe r) :=
  ProgramIO.EquivalentOn.of_implementsOn (PublicIO.implements M encode observe r)
    (candidate_implements M encode observe r)

#axiomsClean candidate_implements
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RopeTransform.PairIO

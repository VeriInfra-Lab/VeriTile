import bench.repairs.tritonbench_g.rope_backward_transform.PairIO
import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.IOBindings

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.PairIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
variable {A : Type} [Inhabited A]

theorem invocation_run (op : Operations A) (r : Request) (row : Nat) (s : State A) :
    (candidateInvocation op r row).run s = (PublicIO.invocation op r row).run s :=
  (IndexCSE.execution op.backend (parameters r) ((PublicIO.invocation op r row).initial s)).symm

theorem invocation_safe (op : Operations A) (r : Request) (row : Nat)
    (bounds : RegionName → Nat) (s : State A) :
    (PublicIO.invocation op r row).Safe bounds s ↔
      (candidateInvocation op r row).Safe bounds s :=
  IndexCSE.trace_safe op.backend (parameters r) bounds ((PublicIO.invocation op r row).initial s)

theorem schedule_run (op : Operations A) (r : Request) (rows : List Nat) (s : State A) :
    runSchedule (rows.map (candidateInvocation op r)) s =
      runSchedule (rows.map (PublicIO.invocation op r)) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    exact congrArg (fun f => (PublicIO.invocation op r row).run s >>= f) (funext ih)

theorem schedule_safe (op : Operations A) (r : Request) (rows : List Nat)
    (bounds : RegionName → Nat) (s : State A) :
    ScheduleSafe bounds (rows.map (PublicIO.invocation op r)) s ↔
      ScheduleSafe bounds (rows.map (candidateInvocation op r)) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rows ih =>
    simp only [List.map_cons,ScheduleSafe,invocation_run]
    exact and_congr (invocation_safe op r row bounds s)
      (forall_congr' (fun t => imp_congr_right (fun _ => ih t)))

/-- Every candidate result read, including odd unwritten tails, realizes the
same ordered scalar formula; safety, successful execution and raw frame hold
for the complete original grid and the original all-payload input domain. -/
theorem candidate_implements (op : Operations A) (project : A → A) (r : Request) :
    (candidate op project r).toReadProgramIO.Implements
      (fun xs => PublicIO.result (fun key row head col =>
        project (PublicIO.logicalResult op r xs key row head col))) := by
  intro xs admitted bounds backed initial inputs
  obtain ⟨safe,final,ran,outputs,frame⟩ := PublicIO.implements op project r xs admitted bounds backed initial inputs
  refine ⟨?_,final,?_,outputs,frame⟩
  · exact (schedule_safe op r (KernelContract.launchRows r).reverse bounds initial).mp safe
  · exact (schedule_run op r (KernelContract.launchRows r).reverse initial).trans ran

/-- Equality of both actual complete schedules at every result read, preserving
both exact writable-cell frames without adding numerical rewrite assumptions. -/
theorem equivalent (op : Operations A) (project : A → A) (r : Request) :
    (PublicIO.sourceIO op project r).Equivalent (candidate op project r) :=
  ReadProgramIO.EquivalentOn.of_implementsOn (PublicIO.implements op project r)
    (candidate_implements op project r)

#axiomsClean candidate_implements
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.PairIO

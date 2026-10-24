import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.AddressIO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.AddressSchedule
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.IOGrid
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.AddressIO
open VeriTile Triton FP.Structural
noncomputable section
variable {A B : Type} [Inhabited A] [Inhabited B]

theorem invocation_run (M : Algebra B) (c : Host.Call) (j : Host.Job) (s : State B) :
    (invocation M c j).run s = (PublicIO.invocation M c j).run s :=
  AddressSchedule.execution_eq M (Host.parameters c) ((PublicIO.invocation M c j).initial s)
    (by change 0<64; decide)

theorem invocation_safe (M : Algebra B) (c : Host.Call) (j : Host.Job)
    (bounds : RegionName → Nat) (s : State B) :
    (invocation M c j).Safe bounds s ↔ (PublicIO.invocation M c j).Safe bounds s :=
  AddressSchedule.safety_iff M (Host.parameters c) ((PublicIO.invocation M c j).initial s)
    bounds (by change 0<64; decide)

theorem schedule_run (M : Host.Job → Algebra B) (c : Host.Call) (work : List Host.Job) (s : State B) :
    runSchedule (work.map (fun j => invocation (M j) c j)) s =
      runSchedule (work.map (fun j => PublicIO.invocation (M j) c j)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons j rest ih =>
    simp only [List.map_cons,runSchedule,invocation_run]
    cases (PublicIO.invocation (M j) c j).run s <;> simp [ih]

theorem schedule_safe (M : Host.Job → Algebra B) (c : Host.Call) (work : List Host.Job)
    (bounds : RegionName → Nat) (s : State B) :
    ScheduleSafe bounds (work.map (fun j => invocation (M j) c j)) s ↔
      ScheduleSafe bounds (work.map (fun j => PublicIO.invocation (M j) c j)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons j rest ih => simp only [List.map_cons,ScheduleSafe,invocation_safe,invocation_run,ih]

/-- The original full logical metadata/read binding and every complete workset
permutation are retained. The pair concludes dual safety, successful runs,
complete typed FP32 outputs and both exterior memory frames. -/
theorem equivalent (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
    (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
    (counts : Fin c.batch → Nat) (work : List Host.Job) (complete : work.Perm (Host.workset c)) :
    (PublicIO.sourceIO M encode c observe counts work).EquivalentOn
      (variant M encode c observe counts work) PublicIO.LengthsMatch := by
  intro xs hm bounds hb initial bound
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    PublicIO.ordered_implements M encode c observe counts work complete xs hm bounds hb initial bound
  refine ⟨safe,?_,final,final,ran,?_,?_,frame,frame⟩
  · exact (schedule_safe M c work bounds initial).mpr safe
  · exact (schedule_run M c work initial).trans ran
  · intro o j active
    rfl

#axiomsClean invocation_run
#axiomsClean invocation_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.AddressIO

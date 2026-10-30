import bench.tritonbench_g.fast_rms_layernorm.IO
import bench.tritonbench_g.fast_rms_layernorm.Proofs.SourcePair
import bench.tritonbench_g.fast_rms_layernorm.Proofs.NativePublicIO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.IOPair
open VeriTile Triton FP.Structural FP.ProfiledRewrite NativeGraph
noncomputable section

theorem forward_run (M : Algebra Term) (h : Call) (epsilon : Term) (pid : Nat) (s : State Term) :
    (NativeIO.forwardJob M h epsilon pid).run s = (forwardJob M h epsilon pid).run s := by
  exact SourcePair.source_exec (engine M) _ h Bool.false

theorem forward_safe (M : Algebra Term) (h : Call) (epsilon : Term) (pid : Nat)
    (s : State Term) (bounds : RegionName → Nat) :
    (NativeIO.forwardJob M h epsilon pid).Safe bounds s ↔ (forwardJob M h epsilon pid).Safe bounds s := by
  exact SourcePair.source_safe (engine M) _ bounds h Bool.false

theorem backward_run (M : Algebra Term) (h : Call) (weight : Bool) (pid count : Nat) (s : State Term) :
    (NativeIO.backwardJob M h weight pid count).run s = (backwardJob M h weight pid count).run s := by
  cases weight
  · exact SourcePair.source_exec (engine M) _ h Bool.true
  · rfl

theorem backward_safe (M : Algebra Term) (h : Call) (weight : Bool) (pid count : Nat)
    (s : State Term) (bounds : RegionName → Nat) :
    (NativeIO.backwardJob M h weight pid count).Safe bounds s ↔ (backwardJob M h weight pid count).Safe bounds s := by
  cases weight
  · exact SourcePair.source_safe (engine M) _ bounds h Bool.true
  · exact Iff.rfl

private theorem map_pair {ι : Type} (left right : ι → Invocation Term)
    (same : ∀ i s, (left i).run s = (right i).run s)
    (safe : ∀ i s bounds, (left i).Safe bounds s ↔ (right i).Safe bounds s)
    (ids : List ι) (s : State Term) :
    runSchedule (ids.map left) s = runSchedule (ids.map right) s ∧
    ∀ bounds, ScheduleSafe bounds (ids.map left) s ↔ ScheduleSafe bounds (ids.map right) s := by
  induction ids generalizing s with
  | nil => exact ⟨rfl,fun _ => Iff.rfl⟩
  | cons i ids ih =>
    constructor
    · simp only [List.map_cons,runSchedule,same]
      congr 1
      funext t
      exact (ih t).1
    · intro bounds
      simp only [List.map_cons,ScheduleSafe,safe,same]
      exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => (ih t).2 bounds)

theorem schedules (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) (s : State Term) :
    runSchedule (NativeIO.schedule R h backward epsilon) s = runSchedule (schedule R h backward epsilon) s ∧
    ∀ bounds, ScheduleSafe bounds (NativeIO.schedule R h backward epsilon) s ↔
      ScheduleSafe bounds (schedule R h backward epsilon) s := by
  cases backward
  · exact map_pair _ _ (fun i => forward_run _ h epsilon i.val)
      (fun i => forward_safe _ h epsilon i.val) (List.finRange h.rows) s
  · by_cases empty : h.rows=0
    · simp [NativeIO.schedule,schedule,empty,runSchedule,ScheduleSafe]
    · have weights : ∀ pid, backwardJob (algebra R Stage.weight) h Bool.true pid (Geometry.featureCount h) =
          NativeIO.backwardJob (algebra R Stage.weight) h Bool.true pid (Geometry.featureCount h) := fun _ => rfl
      have tail := map_pair
        (fun i : Fin h.rows => NativeIO.backwardJob (algebra R .input) h Bool.false i.val h.rows)
        (fun i : Fin h.rows => backwardJob (algebra R .input) h Bool.false i.val h.rows)
        (fun i => backward_run _ h Bool.false i.val h.rows)
        (fun i => backward_safe _ h Bool.false i.val h.rows) (List.finRange h.rows)
      simp only [NativeIO.schedule,schedule,Bool.true_eq_false,↓reduceIte,empty,weights,
        runSchedule_append,scheduleSafe_append]
      constructor
      · congr 1
        funext t
        exact (tail t).1
      · intro bounds
        exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => (tail t).2 bounds)

theorem implemented (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) :
    (NativeIO.program R h backward epsilon).ImplementsOn (Reference.nativeDomain h backward)
      (Reference.native R h backward epsilon) := by
  cases backward
  · intro values _ bounds hb s hin
    exact NativePublicIO.forward R h epsilon values trivial bounds hb s hin
  · exact NativePublicIO.backward R h epsilon

/-- The original complete IO certificate supplies success and frame. The
complete source equality transports exactly that final state to the candidate. -/
theorem equivalent (R : Reductions) (h : Call) (backward : Bool) (epsilon : Term) :
    (NativeIO.program R h backward epsilon).EquivalentOn (variant R h backward epsilon)
      (Reference.nativeDomain h backward) := by
  intro values domain bounds hb s hin
  obtain ⟨safe,final,ran,_,frame⟩ := implemented R h backward epsilon values domain bounds hb s hin
  have same := schedules R h backward epsilon s
  refine ⟨safe,(same.2 bounds).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · exact same.1.symm.trans ran
  · intro o j active
    rfl
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.IOPair

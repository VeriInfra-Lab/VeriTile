import bench.tritonbench_g.kldiv_ops.Proofs.IOSchedule
import bench.tritonbench_g.kldiv_ops.Proofs.SourcePair

namespace VeriTile.Bench.Repairs.KldivOps.Interface
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 24000
variable {α β : Type} [Inhabited α] [Inhabited β]

@[simp] theorem forward_candidate_run (M : Algebra β) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : β) (row : Nat) (s : State β) :
    (forwardCandidateInvocation M p flag red eps row).run s =
      (forwardInvocation M p flag red eps row).run s :=
  SourcePair.forward_run M p flag red _

@[simp] theorem backward_candidate_run (M : Algebra β) (p : Parameters) (flag : Bool)
    (row : Nat) (s : State β) :
    (backwardCandidateInvocation M p flag row).run s = (backwardInvocation M p flag row).run s :=
  SourcePair.backward_run M p flag _

theorem forward_candidate_schedule (M : Algebra β) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : β) (rows : List Nat) (s : State β) :
    runSchedule (rows.map (forwardCandidateInvocation M p flag red eps)) s =
      runSchedule (rows.map (forwardInvocation M p flag red eps)) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih => simp only [List.map_cons,runSchedule,forward_candidate_run]; congr 1; funext t; exact ih t

theorem backward_candidate_schedule (M : Algebra β) (p : Parameters) (flag : Bool)
    (rows : List Nat) (s : State β) :
    runSchedule (rows.map (backwardCandidateInvocation M p flag)) s =
      runSchedule (rows.map (backwardInvocation M p flag)) s := by
  induction rows generalizing s with
  | nil => rfl
  | cons row rest ih => simp only [List.map_cons,runSchedule,backward_candidate_run]; congr 1; funext t; exact ih t

theorem forward_candidate_safety (M : Algebra β) (p : Parameters) (flag : Bool)
    (red : Nat) (eps : β) (rows : List Nat) (bounds : RegionName → Nat) (s : State β)
    (safe : ScheduleSafe bounds (rows.map (forwardInvocation M p flag red eps)) s) :
    ScheduleSafe bounds (rows.map (forwardCandidateInvocation M p flag red eps)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨(SourcePair.forward_safe M p flag red bounds _).mpr safe.1, ?_⟩
    intro t ran
    apply ih
    exact safe.2 t ((forward_candidate_run M p flag red eps row s).symm.trans ran)

theorem backward_candidate_safety (M : Algebra β) (p : Parameters) (flag : Bool)
    (rows : List Nat) (bounds : RegionName → Nat) (s : State β)
    (safe : ScheduleSafe bounds (rows.map (backwardInvocation M p flag)) s) :
    ScheduleSafe bounds (rows.map (backwardCandidateInvocation M p flag)) s := by
  induction rows generalizing s with
  | nil => trivial
  | cons row rest ih =>
    refine ⟨(SourcePair.backward_safe M p flag bounds _).mpr safe.1, ?_⟩
    intro t ran
    apply ih
    exact safe.2 t ((backward_candidate_run M p flag row s).symm.trans ran)

/-- All rows/chunks, both target branches, all reduction codes, typed output
cells, both source traces and complete exterior frames. -/
theorem forward_equivalence (M : Algebra β) (p : Parameters) (flag : Bool) (red : Nat)
    (eps : β) (positive : 0<p.block) (hy : p.Y≠p.O) (ht : p.T≠p.O)
    (encode : α → β) (observe : β → α) :
    (forwardIO M p flag red eps encode observe).Equivalent
      (forwardCandidate M p flag red eps encode observe) := by
  intro vals _ bounds hb initial _
  have safe : ScheduleSafe bounds ((forwardIO M p flag red eps encode observe).schedule vals) initial := by
    apply forward_schedule_safe M p flag red eps positive (List.range p.rows)
      (fun row member => List.mem_range.mp member) bounds
    · intro i hi
      exact hb.2.1 ⟨0,by simp [forwardIO,forwardMemory,forwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.1 ⟨1,by simp [forwardIO,forwardMemory,forwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 ⟨0,by simp [forwardIO,forwardMemory,outputChannels]⟩ (⟨i,hi⟩,PUnit.unit) trivial
  refine ⟨safe,forward_candidate_safety M p flag red eps _ bounds initial safe,?_⟩
  obtain ⟨final,ran,_,frame⟩ := KernelResults.forward M p flag red eps initial positive hy ht
  have framed : (forwardIO M p flag red eps encode observe).Frame vals initial final := by
    intro r i outside
    apply frame r i
    by_cases different : r≠p.O
    · exact .inl different
    · have same : r=p.O := Classical.not_not.mp different
      subst r
      right
      by_contra h
      have bound : i<outputLength p red := Nat.lt_of_not_ge h
      have no := outside.1 ⟨0,by simp [forwardIO,forwardMemory,outputChannels]⟩ (⟨i,bound⟩,PUnit.unit) trivial
      exact no.elim (fun bad => bad rfl) (fun bad => bad rfl)
  have executed : runSchedule ((forwardIO M p flag red eps encode observe).schedule vals) initial=some final :=
    (forward_schedule_run M p flag red eps _ initial).trans ran
  refine ⟨final,final,executed,?_,fun _ _ _ => rfl,framed,framed⟩
  exact (forward_candidate_schedule M p flag red eps _ initial).trans executed

theorem backward_equivalence (M : Algebra β) (p : Parameters) (flag : Bool)
    (positive : 0<p.block) (ht : p.T≠p.O) (encode : α → β) (observe : β → α) :
    (backwardIO M p flag encode observe).Equivalent (backwardCandidate M p flag encode observe) := by
  intro vals _ bounds hb initial _
  have safe : ScheduleSafe bounds ((backwardIO M p flag encode observe).schedule vals) initial := by
    apply backward_schedule_safe M p flag positive (List.range p.rows)
      (fun row member => List.mem_range.mp member) bounds
    · intro i hi
      exact hb.2.1 ⟨0,by simp [backwardIO,backwardMemory,backwardInputs]⟩ (⟨i,hi⟩,PUnit.unit) trivial
    · intro i hi
      exact hb.2.2.1 ⟨0,by simp [backwardIO,backwardMemory,outputChannels]⟩ (⟨i,hi⟩,PUnit.unit) trivial
  refine ⟨safe,backward_candidate_safety M p flag _ bounds initial safe,?_⟩
  obtain ⟨final,ran,_,frame⟩ := KernelResults.backward M p flag initial positive ht
  have framed : (backwardIO M p flag encode observe).Frame vals initial final := by
    intro r i outside
    apply frame r i
    by_cases different : r≠p.O
    · exact .inl different
    · have same : r=p.O := Classical.not_not.mp different
      subst r
      right
      by_contra h
      have bound : i<p.rows*p.cols := Nat.lt_of_not_ge h
      have no := outside.1 ⟨0,by simp [backwardIO,backwardMemory,outputChannels]⟩ (⟨i,bound⟩,PUnit.unit) trivial
      exact no.elim (fun bad => bad rfl) (fun bad => bad rfl)
  have executed : runSchedule ((backwardIO M p flag encode observe).schedule vals) initial=some final :=
    (backward_schedule_run M p flag _ initial).trans ran
  refine ⟨final,final,executed,?_,fun _ _ _ => rfl,framed,framed⟩
  exact (backward_candidate_schedule M p flag _ initial).trans executed

#axiomsClean forward_equivalence
#axiomsClean backward_equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Interface

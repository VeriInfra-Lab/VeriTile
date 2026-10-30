import bench.tritonbench_g.softmax_reducev.Proofs.WholeHost
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Host
open VeriTile.Triton FP.Structural
open SoftmaxTriton1 (Result)
noncomputable section
set_option maxHeartbeats 2000000

/-- This helper law is discharged from the actual full typed source at each
public Real/native entry; it is never a public numerical assumption. -/
structure SourceLawGeneral {α} (c : Call)
    (evaluate : Job → (Nat → α) → Option (Nat → α)) (target : Job → Nat → α) : Prop where
  success : ∀ j s t, evaluate j s=some t →
    (∀ d, d < c.dimensions → t (address c j d)=target j d) ∧ Frame c [j] s t

/-- Same ordinary local launch relation, now usable after observing physical
FP32 graph replicas. No result formula appears in either runtime field. -/
structure RuntimeGeneral {α} (c : Call)
    (evaluate : Job → (Nat → α) → Option (Nat → α))
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) : Prop where
  success : ∀ j s t, Legal c j → invoke j s=.ok () t → evaluate j s=some t
  error : ∀ j s e t, Legal c j → invoke j s=.error e t → Frame c [j] s t

theorem jobs_return_general {α} [Inhabited α] (c : Call) (evaluate : Job → (Nat → α) → Option (Nat → α)) (target : Job → Nat → α)
    (source : SourceLawGeneral c evaluate target)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : RuntimeGeneral c evaluate invoke)
    (jobs : List Job) (legal : ∀ j∈jobs,Legal c j) (s t : Nat → α)
    (run : runJobs invoke jobs s=.ok () t) :
    (∀ j∈jobs, ∀ d, d < c.dimensions → t (address c j d)=target j d) ∧ Frame c jobs s t := by
  induction jobs generalizing s with
  | nil =>
    simp only [runJobs,Result.ok.injEq] at run
    cases run.2
    exact ⟨by simp,fun _ _ => rfl⟩
  | cons j js ih =>
    cases call : invoke j s with
    | error e u => simp [runJobs,call,Result.bind] at run
    | ok v u =>
      cases v
      have hj := legal j (by simp)
      have first := source.success j s u (rt.success j s u hj call)
      have tailRun : runJobs invoke js u=.ok () t := by simpa [runJobs,call,Result.bind] using run
      have rest := ih (fun k hk => legal k (by simp [hk])) u tailRun
      refine ⟨?_,?_⟩
      · intro k hk d hd
        by_cases tail : k∈js
        · exact rest.1 k tail d hd
        · have kj : k=j := by simpa [tail] using hk
          subst k
          have saved := rest.2 (address c j d) (by
            intro k hk hit
            obtain ⟨e,he,eq⟩ := hit
            have same := ownership c j k hj (legal k (by simp [hk])) d e hd he eq
            exact tail (same.1.symm ▸ hk))
          exact saved.trans (first.1 d hd)
      · intro a outside
        exact (rest.2 a (fun k hk => outside k (by simp [hk]))).trans
          (first.2 a (by intro k hk;exact outside k (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp hk)))))

theorem jobs_all_outcomes_general {α} [Inhabited α] (c : Call) (evaluate : Job → (Nat → α) → Option (Nat → α)) (target : Job → Nat → α)
    (source : SourceLawGeneral c evaluate target)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : RuntimeGeneral c evaluate invoke)
    (jobs : List Job) (legal : ∀ j∈jobs,Legal c j) (s : Nat → α) :
    Frame c jobs s (runJobs invoke jobs s).state := by
  induction jobs generalizing s with
  | nil => exact fun _ _ => rfl
  | cons j js ih =>
    have hj := legal j (by simp)
    cases call : invoke j s with
    | error e u =>
      have first := rt.error j s e u hj call
      intro a outside
      simpa [runJobs,call,Result.bind,Result.state] using
        first a (by intro k hk;exact outside k (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp hk))))
    | ok v u =>
      cases v
      have first := (source.success j s u (rt.success j s u hj call)).2
      have rest := ih (fun k hk => legal k (by simp [hk])) u
      intro a outside
      simpa [runJobs,call,Result.bind] using
        (rest a (fun k hk => outside k (by simp [hk]))).trans
          (first a (by intro k hk;exact outside k (List.mem_cons.mpr (Or.inl (List.mem_singleton.mp hk)))))
theorem whole_general {α} [Inhabited α] (c : Call) (evaluate : Job → (Nat → α) → Option (Nat → α)) (target : Job → Nat → α)
    (source : SourceLawGeneral c evaluate target) (base : Nat)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : RuntimeGeneral c evaluate invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial final : Memory α)
    (run : wrapper c base schedule invoke initial=.ok () final) :
    (∀ j, Legal c j → ∀ d, d < c.dimensions → final "Out" (base+address c j d)=target j d) ∧
    (∀ r a, Outside c base r a → final r a=initial r a) := by
  by_cases empty : noWork c
  · have eq : initial=final := by simpa [wrapper,empty,Result.ok.injEq] using run
    subst final
    refine ⟨?_,fun _ _ _ => rfl⟩
    intro j hj d hd
    rcases empty with e | e <;> dsimp [Legal] at hj <;> omega
  · have run' : runFull base invoke schedule initial=.ok () final := by simpa [wrapper,empty] using run
    rw [runFull_projection] at run'
    cases call : runJobs invoke schedule (readOutput base initial) with
    | error e out => simp [call,liftResult] at run'
    | ok v out =>
      cases v
      have eq : putOutput base initial out=final := by simpa [call,liftResult] using run'
      subst final
      have result := jobs_return_general c evaluate target source invoke rt schedule
        (fun j hj => (mem_workset c j).mp (complete.mem_iff.mp hj)) (readOutput base initial) out call
      refine ⟨?_,?_⟩
      · intro j hj d hd
        simpa [putOutput] using result.1 j (complete.mem_iff.mpr ((mem_workset c j).mpr hj)) d hd
      · intro r a outside
        by_cases h : r="Out" ∧ base ≤ a
        · have off : Off c (workset c) (a-base) := by
            rcases outside with out | before | off
            · exact (out h.1).elim
            · omega
            · exact off
          have same := result.2 (a-base) (fun j hj => off j (complete.mem_iff.mp hj))
          simpa [putOutput,h,readOutput,Nat.add_sub_of_le h.2,h.1] using same
        · simp [putOutput,h]

theorem all_outcomes_general {α} [Inhabited α] (c : Call) (evaluate : Job → (Nat → α) → Option (Nat → α)) (target : Job → Nat → α)
    (source : SourceLawGeneral c evaluate target) (base : Nat)
    (invoke : Job → (Nat → α) → Result (Nat → α) Unit) (rt : RuntimeGeneral c evaluate invoke)
    (schedule : List Job) (complete : schedule.Perm (workset c)) (initial : Memory α)
    (r : RegionName) (a : Nat) (outside : Outside c base r a) :
    (wrapper c base schedule invoke initial).state r a=initial r a := by
  by_cases empty : noWork c
  · simp [wrapper,empty,Result.state]
  · have frame := jobs_all_outcomes_general c evaluate target source invoke rt schedule
      (fun j hj => (mem_workset c j).mp (complete.mem_iff.mp hj)) (readOutput base initial)
    have framed (out : Nat → α) (hframe : Frame c schedule (readOutput base initial) out) :
        putOutput base initial out r a=initial r a := by
      by_cases h : r="Out" ∧ base ≤ a
      · have off : Off c (workset c) (a-base) := by
          rcases outside with out | before | off
          · exact (out h.1).elim
          · omega
          · exact off
        have same := hframe (a-base) (fun j hj => off j (complete.mem_iff.mp hj))
        simpa [putOutput,h,readOutput,Nat.add_sub_of_le h.2,h.1] using same
      · simp [putOutput,h]
    simp only [wrapper,if_neg empty,runFull_projection]
    cases call : runJobs invoke schedule (readOutput base initial) with
    | error e out =>
      simp only [call,Result.state] at frame
      exact framed out frame
    | ok v out =>
      cases v
      simp only [call,Result.state] at frame
      exact framed out frame


#axiomsClean jobs_return_general
#axiomsClean jobs_all_outcomes_general
#axiomsClean whole_general
#axiomsClean all_outcomes_general
end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Host

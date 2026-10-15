import bench.tritonbench_g.quantize_global.FiniteCorrect

namespace QuantizeGlobalFinite
set_option maxRecDepth 24000
set_option maxHeartbeats 8000000

/-- No defined-conversion premise: actual scalar execution either reports an
undefined active conversion or performs the source stores with their values. -/
theorem lanes_outcome (b : Numerics) (d : Storage) (p : Parameters) (xs : Nat → Word d)
    (v : Word (compute d)) (snapshot : Memory) (inputs : Inputs d p xs v snapshot)
    (c : Config) (pid : Nat) (lanes : List Nat) (working : Memory) :
    (∃ i, runLanes b d p snapshot v c pid lanes working=.error (.conversionUndefined i)) ∨
    runLanes b d p snapshot v c pid lanes working=
      .ok (referenceIndices b d p xs v (lanes.map (offset c pid)) working) := by
  induction lanes generalizing working with
  | nil => right; simp [runLanes]
  | cons lane lanes ih =>
    by_cases active : offset c pid lane<p.n
    · cases hz : result b d (xs (offset c pid lane)) v with
      | none =>
        left
        refine ⟨offset c pid lane,?_⟩
        rw [runLanes,if_pos active,inputs.1 _ active,read_same]
        dsimp only
        rw [source_scalar,hz]
      | some z =>
        rcases ih (write working p.output (offset c pid lane) (some (.mk .i8 z))) with hf|hs
        · obtain ⟨i,hi⟩ := hf
          left; refine ⟨i,?_⟩
          rw [runLanes,if_pos active,inputs.1 _ active,read_same]
          dsimp only
          rw [source_scalar,hz]
          exact hi
        · right
          rw [runLanes,if_pos active,inputs.1 _ active,read_same]
          dsimp only
          rw [source_scalar,hz]
          dsimp only
          rw [hs]
          simp [reference_cons,active,outCell,hz]
          rfl
    · rcases ih working with hf|hs
      · obtain ⟨i,hi⟩ := hf
        left; exact ⟨i,by rw [runLanes,if_neg active]; exact hi⟩
      · right
        rw [runLanes,if_neg active,hs]
        simp [reference_cons,active]

theorem grid_outcome (b : Numerics) (d : Storage) (p : Parameters) (xs : Nat → Word d)
    (v : Word (compute d)) (sep : Separated p) (c : Config) (pids : List Nat)
    (initial : Memory) (inputs : Inputs d p xs v initial) :
    (∃ i, runGrid b d p c pids initial=.error (.conversionUndefined i)) ∨
    runGrid b d p c pids initial=.ok (referenceIndices b d p xs v (footprint c pids) initial) := by
  induction pids generalizing initial with
  | nil => right; simp [runGrid,footprint]
  | cons pid pids ih =>
    rcases lanes_outcome b d p xs v initial inputs c pid (List.range c.block) initial with hf|hs
    · obtain ⟨i,hi⟩ := hf
      have hp : runProgram b d p c pid initial=.error (.conversionUndefined i) := by
        rw [runProgram,inputs.2,read_same]; exact hi
      left; exact ⟨i,by rw [runGrid,hp]; rfl⟩
    · have hp : runProgram b d p c pid initial=.ok
          (referenceIndices b d p xs v ((List.range c.block).map (offset c pid)) initial) := by
        rw [runProgram,inputs.2,read_same]; exact hs
      rcases ih _ (reference_inputs b d p xs v _ initial sep inputs) with hf|hs
      · obtain ⟨i,hi⟩ := hf
        left; exact ⟨i,by rw [runGrid,hp]; exact hi⟩
      · right
        rw [runGrid,hp]
        change runGrid b d p c pids _ = _
        rw [hs,reference_append]
        rfl

theorem complete_reference (b : Numerics) (d : Storage) (p : Parameters) (xs : Nat → Word d)
    (v : Word (compute d)) (c : Config) (initial : Memory) :
    referenceIndices b d p xs v (footprint c (List.range (programs p.n c))) initial=
      reference b d p xs v initial := by
  funext r i
  by_cases hi : i<p.n
  · have hc : i∈footprint c (List.range (programs p.n c)) := by
      rw [footprint_membership,List.mem_range]
      cases c <;> simp [programs,Config.block] <;> omega
    simp [referenceIndices,reference,hi,hc]
  · simp [referenceIndices,reference,hi]

theorem trials_outcome (b : Numerics) (d : Storage) (p : Parameters) (xs : Nat → Word d)
    (v : Word (compute d)) (sep : Separated p) (trials : List Config) (last : Config)
    (initial : Memory) (inputs : Inputs d p xs v initial) :
    (∃ i, runTrials b d p (trials++[last]) initial=.error (.conversionUndefined i)) ∨
    runTrials b d p (trials++[last]) initial=.ok (reference b d p xs v initial) := by
  induction trials generalizing initial with
  | nil =>
    rcases grid_outcome b d p xs v sep last (List.range (programs p.n last)) initial inputs with hf|hs
    · obtain ⟨i,hi⟩ := hf
      left; exact ⟨i,by simp only [List.nil_append,runTrials,hi]; rfl⟩
    · right
      rw [complete_reference] at hs
      simp only [List.nil_append,runTrials,hs]
      rfl
  | cons c cs ih =>
    rcases grid_outcome b d p xs v sep c (List.range (programs p.n c)) initial inputs with hf|hs
    · obtain ⟨i,hi⟩ := hf
      left; exact ⟨i,by simp only [List.cons_append,runTrials,hi]; rfl⟩
    · rw [complete_reference] at hs
      rcases ih _ (full_reference_inputs b d p xs v initial sep inputs) with hf|ht
      · obtain ⟨i,hi⟩ := hf
        left; exact ⟨i,by simp only [List.cons_append,runTrials,hs]; exact hi⟩
      · right
        simp only [List.cons_append,runTrials,hs]
        change runTrials b d p (cs++[last]) _ = _
        rw [ht,reference_idempotent]

/-- Full finite numerical-value partition: undefined conversion is an explicit
source observation, never an arbitrary callback value or a proved native trap.
Successful native primitive mechanics remain the stated runtime boundary. -/
theorem quantize_global_all_finite_value_outcomes
    (b : Numerics) (d : Storage) (a : Input) (alloc : Allocations) (m : Memory)
    (x : String) (logical : List (Word d)) (xs : Nat → Word d)
    (D : Domain d a alloc m x logical xs) (nonempty : 0<a.numel)
    (rank : a.shape≠[]) (cuda : a.isCuda=true) (trials : List Config) (last : Config) :
    ∃ maximum, produced b d logical=some maximum ∧
      ((∃ i, executeAPI b (fun _ => none) (dtype d) a alloc m x trials last=
          .attempted d (.error (.interpreter (.conversionUndefined i)))) ∨
       (∃ ret, executeAPI b (fun _ => none) (dtype d) a alloc m x trials last=.attempted d (.ok ret) ∧
          Result b d a alloc m xs maximum ret)) := by
  have hlen : a.logicalOffsets.length=logical.length := List.Forall₂.length_eq D.logical
  have hne : logical≠[] := by
    intro he
    have hz : a.logicalOffsets.length=0 := by simpa only [he,List.length_nil] using hlen
    have hc := D.metadataCount
    omega
  cases logical with
  | nil => exact (hne rfl).elim
  | cons h t =>
    let maximum := b.maximum d (b.abs d h) (t.map (b.abs d))
    have hm : produced b d (h::t)=some maximum := rfl
    refine ⟨maximum,hm,?_⟩
    have hp := prepare_success b d a alloc m x (h::t) maximum D.logical hm rank cuda
    have hin := allocated_inputs d a alloc m x maximum (b.reciprocal d maximum) xs D.fresh D.linear
    have hsep : Separated (parameters a alloc x) := ⟨D.fresh.2.2.1,D.fresh.2.2.2.2.2⟩
    have hroute := admitted_route d a.numel nonempty
    have hz : a.shape.isEmpty=false := by cases hs : a.shape <;> simp_all
    rcases trials_outcome b d (parameters a alloc x) xs (b.reciprocal d maximum) hsep trials last _ hin with hf|ht
    · obtain ⟨i,hi⟩ := hf
      left; refine ⟨i,?_⟩
      simp [executeAPI,cuda,hz,hroute,Nat.not_lt_of_ge D.indexBound,Nat.not_lt_of_ge D.linearBounds,
        execute,hp,hi]
    · right
      refine ⟨⟨reference b d (parameters a alloc x) xs (b.reciprocal d maximum)
          (allocated d a alloc m maximum (b.reciprocal d maximum)),
          (outputTensor a alloc,maximumTensor d a alloc)⟩,?_,?_⟩
      · simp [executeAPI,cuda,hz,hroute,Nat.not_lt_of_ge D.indexBound,Nat.not_lt_of_ge D.linearBounds,
          execute,hp,ht]
      · refine ⟨rfl,?_,?_,?_,?_⟩
        · intro i hi; simp [reference,parameters,hi]
        · simp [reference,parameters,D.fresh.2.2.2.2.1,allocated]
        · simp [reference,parameters,D.fresh.2.2.2.2.2,allocated,Ne.symm D.fresh.2.2.2.1]
        · intro r i ho hm hs
          simp [reference,parameters,ho,allocated,hm,hs]

#axiomsClean lanes_outcome
#axiomsClean grid_outcome
#axiomsClean trials_outcome
#axiomsClean quantize_global_all_finite_value_outcomes
#guard_msgs (drop info) in
#auditModuleAxioms
end QuantizeGlobalFinite

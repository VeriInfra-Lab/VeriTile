import bench.tritonbench_g.softmax_flaggems.Proofs.HostGridDispatch
import bench.tritonbench_g.softmax_flaggems.Proofs.SourceSupport

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000

instance (storage : Storage) (one : Bool) : Decidable (SourceProfileSupported storage one) :=
  inferInstanceAs (Decidable (one=Bool.true ∨ storage≠.fp64))

/-- The list records every actual benchmark kernel invocation, including warmup
and timing repetitions. The empty list represents a cached selection. Each
invocation receives the preceding mutable state. -/
noncomputable def trialRun (storage : Storage) (P G Y : RegionName) (d : Dimensions) :
    List (Fin 6) → BlockState → Option BlockState
  | [],s => some s
  | choice::rest,s => do
      let t ← backwardRun storage P G Y d (backwardConfig d choice) s
      trialRun storage P G Y d rest t

def TrialsSupported (storage : Storage) (d : Dimensions) (trials : List (Fin 6)) : Prop :=
  ∀ choice ∈ trials, SourceProfileSupported storage (backwardConfig d choice).one

structure TrialError where
  choice : Fin 6
  compilerRejected : Bool
  beforeFailure : BlockState

/-- Source frontend admission is checked before executing each trial. Rejection
retains the state after all preceding launches and the first failing choice. -/
noncomputable def checkedTrialRun (storage : Storage) (P G Y : RegionName) (d : Dimensions) :
    List (Fin 6) → BlockState → Except TrialError BlockState
  | [],s => .ok s
  | choice::rest,s =>
      if SourceProfileSupported storage (backwardConfig d choice).one then
        match backwardRun storage P G Y d (backwardConfig d choice) s with
        | none => .error ⟨choice,Bool.false,s⟩
        | some t => checkedTrialRun storage P G Y d rest t
      else .error ⟨choice,Bool.true,s⟩

theorem checkedTrialRun_success (storage : Storage) (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (hs : TrialsSupported storage d trials)
    (s t : BlockState) (he : trialRun storage P G Y d trials s=some t) :
    checkedTrialRun storage P G Y d trials s=.ok t := by
  induction trials generalizing s with
  | nil =>
      have h : s=t := by simpa [trialRun] using he
      simpa [checkedTrialRun,h]
  | cons c cs ih =>
      have hc := hs c (by simp)
      have hcs : TrialsSupported storage d cs := fun c h ↦ hs c (by simp [h])
      cases hb : backwardRun storage P G Y d (backwardConfig d c) s with
      | none => simp [trialRun,hb] at he
      | some mid =>
          simp only [trialRun,hb,bind,Option.bind] at he
          simpa [checkedTrialRun,hc,hb] using ih hcs mid he

theorem checkedTrialRun_reject_head (storage : Storage) (P G Y : RegionName) (d : Dimensions)
    (c : Fin 6) (cs : List (Fin 6)) (s : BlockState)
    (h : ¬SourceProfileSupported storage (backwardConfig d c).one) :
    checkedTrialRun storage P G Y d (c::cs) s=.error ⟨c,Bool.true,s⟩ := by
  simp [checkedTrialRun,h]

theorem checkedTrialRun_first_rejection (storage : Storage) (P G Y : RegionName) (d : Dimensions)
    (prior : List (Fin 6)) (c : Fin 6) (rest : List (Fin 6))
    (hs : TrialsSupported storage d prior) (s t : BlockState)
    (he : trialRun storage P G Y d prior s=some t)
    (hr : ¬SourceProfileSupported storage (backwardConfig d c).one) :
    checkedTrialRun storage P G Y d (prior++c::rest) s=.error ⟨c,Bool.true,t⟩ := by
  induction prior generalizing s with
  | nil =>
      have h : s=t := by simpa [trialRun] using he
      subst s
      exact checkedTrialRun_reject_head storage P G Y d c rest t hr
  | cons q qs ih =>
      have hq := hs q (by simp)
      have hqs : TrialsSupported storage d qs := fun q h ↦ hs q (by simp [h])
      cases hb : backwardRun storage P G Y d (backwardConfig d q) s with
      | none => simp [trialRun,hb] at he
      | some mid =>
          simp only [trialRun,hb,bind,Option.bind] at he
          simpa [checkedTrialRun,hq,hb] using ih hqs mid he

theorem trialRun_frame (storage : Storage) (P G Y : RegionName) (d : Dimensions)
    (hN : 0<d.N) (hK : 0<d.K) (hPY : P≠Y) (hGY : G≠Y)
    (trials : List (Fin 6)) (s : BlockState) :
    ∃ t, trialRun storage P G Y d trials s=some t ∧
      BlockState.WriteWithin (fullSlice Y (d.M*d.N*d.K)) s t ∧ t.undef=s.undef := by
  induction trials generalizing s with
  | nil => exact ⟨s,rfl,fun _ _ _ ↦ rfl,rfl⟩
  | cons c cs ih =>
      obtain ⟨mid,hb,hf,_,hu⟩ := backward_dispatch storage s P G Y d (backwardConfig d c)
        hN hK (backwardConfig_positive d c).1 (backwardConfig_positive d c).2
        (backwardConfig_one d c) hPY hGY
      obtain ⟨t,ht,hft,hut⟩ := ih mid
      refine ⟨t,by simp [trialRun,hb,ht],?_,hut.trans hu⟩
      intro r o h
      exact (hf r o h).trans (hft r o h)

theorem backwardValue_stable (s t : BlockState) (P G Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hPY : P≠Y) (hGY : G≠Y)
    (hf : BlockState.WriteWithin (fullSlice Y (d.M*d.N*d.K)) s t)
    (hu : t.undef=s.undef) : backwardValue t P G d c=backwardValue s P G d c := by
  have hp : t.mem P=s.mem P := funext fun o ↦ (hf P o (by simp [fullSlice,hPY])).symm
  have hg : t.mem G=s.mem G := funext fun o ↦ (hf G o (by simp [fullSlice,hGY])).symm
  funext a
  unfold backwardValue
  split_ifs <;> simp only [backwardInnerValue,backwardNonInnerValue,
    RealBackwardInner.expected,RealBackwardNonInner.expected,RealMath.backwardPadded,
    RealBackwardInner.sample,RealBackwardNonInner.sample,BlockState.readMem,hp,hg,hu]

noncomputable def tunedBackward (storage : Storage) (P G Y : RegionName) (d : Dimensions)
    (trials : List (Fin 6)) (choice : Fin 6) (s : BlockState) : Option BlockState := do
  let warmed ← trialRun storage P G Y d trials s
  backwardRun storage P G Y d (backwardConfig d choice) warmed

theorem tuned_backward_dispatch (storage : Storage) (base : BlockState) (P G Y : RegionName)
    (d : Dimensions) (trials : List (Fin 6)) (choice : Fin 6)
    (hN : 0<d.N) (hK : 0<d.K) (hPY : P≠Y) (hGY : G≠Y) :
    ∃ final, tunedBackward storage P G Y d trials choice base=some final ∧
      FlatResult Y (d.M*d.N*d.K) (backwardValue base P G d (backwardConfig d choice)) base final := by
  obtain ⟨warm,hw,hfw,huw⟩ := trialRun_frame storage P G Y d hN hK hPY hGY trials base
  obtain ⟨final,hb,hfb,hv,hu⟩ := backward_dispatch storage warm P G Y d (backwardConfig d choice)
    hN hK (backwardConfig_positive d choice).1 (backwardConfig_positive d choice).2
    (backwardConfig_one d choice) hPY hGY
  refine ⟨final,by simp [tunedBackward,hw,hb],?_,?_,hu.trans huw⟩
  · intro r o h
    exact (hfw r o h).trans (hfb r o h)
  · intro o ho
    rw [← backwardValue_stable base warm P G Y d (backwardConfig d choice) hPY hGY hfw huw]
    exact hv o ho

theorem forward_source_domain (waves : WavesAboveOne) (d : Dimensions) (sms : Nat) :
    ForwardSourceDomain (forwardConfig waves d sms).one d.N (forwardConfig waves d sms).B := by
  intro h
  rw [forwardConfig_one,oneTile] at h
  have hn : ¬ d.N≤(forwardConfig waves d sms).B := by simpa using h
  omega

#axiomsClean checkedTrialRun_success
#axiomsClean checkedTrialRun_reject_head
#axiomsClean checkedTrialRun_first_rejection
#axiomsClean trialRun_frame
#axiomsClean backwardValue_stable
#axiomsClean tuned_backward_dispatch
#axiomsClean forward_source_domain
#specNonCircular backwardValue avoiding [trialRun,checkedTrialRun,tunedBackward,backwardRun,run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid

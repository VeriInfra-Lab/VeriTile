import bench.tritonbench_g.spinning_lock_reduction.Proofs.PhasedControl

noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Phased
open VeriTile.Bench.TritonBenchG.SpinningLock Wide

structure Plan where
  producers : List Nat
  consumers : List Nat

def flatten (h : HostArgs) (plan : Plan) : List Nat :=
  activeSchedule h .producers plan.producers ++ activeSchedule h .consumers plan.consumers

def producerState (b : Backend V) (h : HostArgs) (plan : Plan) (mem : Memory V) :=
  (Wide.run b h (activeSchedule h .producers plan.producers) (Wide.initial b mem)).2

def finalState (b : Backend V) (h : HostArgs) (plan : Plan) (mem : Memory V) :=
  (Wide.run b h (flatten h plan) (Wide.initial b mem)).2

/-- A finite execution observation, separate from the wrapper's host return.
The two lists are per-stage interleavings of actual active program steps. -/
def observe (b : Backend V) (encoding : SignEncoding V) (flags : ViewFlags)
    (h : HostArgs) (plan : Plan) (raw : Memory V) : WrapperResult V :=
  physicalLaunch b encoding flags h (flatten h plan) raw

theorem run_append (b : Backend V) (h : HostArgs) (xs ys : List Nat) (s : Wide.State V) :
    Wide.run b h (xs ++ ys) s =
      ((Wide.run b h xs s).1 ++ (Wide.run b h ys (Wide.run b h xs s).2).1,
       (Wide.run b h ys (Wide.run b h xs s).2).2) := by
  induction xs generalizing s with
  | nil => rfl
  | cons p ps ih => simp only [Wide.run, List.cons_append, ih]

theorem two_source_invocations (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (plan : Plan) (mem : Memory V) :
    let first := stageRun b h .producers plan.producers (Wide.initial b mem)
    let second := stageRun b h .consumers plan.consumers first.2
    Wide.launch b h (flatten h plan) mem = ⟨(), first.1 ++ second.1, second.2⟩ := by
  dsimp only
  rw [stageRun_eq b h buf _ _ _ (by intro p; trivial)]
  rw [stageRun_eq b h buf _ _ _ (run_controlBound h b _ _ (by intro p; trivial))]
  rw [launch_eq_run b h buf, flatten, run_append]

theorem run_untouched_control (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (p : Nat) (hp : p ∉ xs) : (Wide.run b h xs s).2.pc p = s.pc p := by
  induction xs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    have hpq : p ≠ q := by intro he; exact hp (List.mem_cons.mpr (Or.inl he))
    have hps : p ∉ qs := fun he => hp (List.mem_cons_of_mem _ he)
    rw [show (Wide.run b h (q :: qs) s).2 = (Wide.run b h qs (Wide.step b h q s).2).2 from rfl,
      ih _ hps]
    simp [Wide.step, put, hpq]

theorem inactive_not_mem (h : HostArgs) (stage : Stage) (xs : List Nat) (p : Nat)
    (hp : selected h stage p = false) : p ∉ activeSchedule h stage xs := by
  simp [activeSchedule, hp]

theorem producer_leader_control_fresh (b : Backend V) (h : HostArgs) (plan : Plan)
    (mem : Memory V) (p : Nat) (red : reducer h p = true) :
    (producerState b h plan mem).pc p = ⟨1, .branch⟩ := by
  unfold producerState
  rw [run_untouched_control b h _ _ p (inactive_not_mem h .producers _ p (by simp [selected, red]))]
  rfl

theorem step_other_acc (b : Backend V) (h : HostArgs) (s : Wide.State V)
    (p q lane : Nat) (hpq : p ≠ q) : (Wide.step b h q s).2.values.acc p lane = s.values.acc p lane := by
  by_cases hg : q ≥ h.numSms
  · simp [Wide.step, Wide.decode, hg, execute]
  · cases hc : (s.pc q).phase <;> simp only [Wide.step, Wide.decode, if_neg hg, hc]
    all_goals try split_ifs
    all_goals simp [execute, put, hpq]

theorem run_untouched_acc (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (p lane : Nat) (hp : p ∉ xs) :
    (Wide.run b h xs s).2.values.acc p lane = s.values.acc p lane := by
  induction xs generalizing s with
  | nil => rfl
  | cons q qs ih =>
    have hpq : p ≠ q := by intro he; exact hp (List.mem_cons.mpr (Or.inl he))
    have hps : p ∉ qs := fun he => hp (List.mem_cons_of_mem _ he)
    exact (ih _ hps).trans (step_other_acc b h s p q lane hpq)

/-- The fresh consumer accumulator is exactly zero: no producer-phase step
can modify a consumer's local slot. Retaining the ghost slot across the two
invocations therefore agrees with the actual second invocation's tl.zeros. -/
theorem producer_leader_acc_fresh (b : Backend V) (h : HostArgs) (plan : Plan)
    (mem : Memory V) (p lane : Nat) (red : reducer h p = true) :
    (producerState b h plan mem).values.acc p lane = b.zero := by
  unfold producerState
  rw [run_untouched_acc b h _ _ p lane
    (inactive_not_mem h .producers _ p (by simp [selected, red]))]
  rfl

/-- Ordinary same-stream execution ordering. This is a boundary on when the
second invocation may start, not a postulate of payload or numerical results.
It also admits arbitrary producer-only prefixes while the first is pending. -/
def StreamPrefix (b : Backend V) (h : HostArgs) (plan : Plan) (mem : Memory V) : Prop :=
  activeSchedule h .consumers plan.consumers ≠ [] →
    StageDone h .producers (producerState b h plan mem)

structure Synchronized (b : Backend V) (h : HostArgs) (plan : Plan) (mem : Memory V) : Prop where
  producers : StageDone h .producers (producerState b h plan mem)
  consumers : StageDone h .consumers (finalState b h plan mem)

theorem Synchronized.stream (sync : Synchronized b h plan mem) : StreamPrefix b h plan mem :=
  fun _ => sync.producers

theorem Synchronized.all_done (sync : Synchronized b h plan mem) :
    Wide.allDone h (finalState b h plan mem) = true := by
  apply List.all_eq_true.mpr
  intro p hp
  have hps := List.mem_range.mp hp
  apply beq_iff_eq.mpr
  by_cases red : reducer h p = true
  · exact sync.consumers p hps red
  · have rf : reducer h p = false := Bool.eq_false_iff.mpr red
    have hd := sync.producers p hps (by simp [selected, rf])
    unfold finalState flatten
    rw [Wide.run_append_state]
    rw [run_untouched_control b h _ _ p (inactive_not_mem h .consumers _ p rf)]
    exact hd

theorem synchronized_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (encoding : SignEncoding V) (flags : ViewFlags) (h : HostArgs) (buf : Buffers h)
    (plan : Plan) (raw : Memory V)
    (ready : PublishedReady b h (interpretMemory encoding flags raw))
    (sync : Synchronized b h plan (interpretMemory encoding flags raw)) :
    CompleteFacts b h buf (flatten h plan) (interpretMemory encoding flags raw)
      (interpretResult encoding flags (observe b encoding flags h plan raw)) := by
  simp only [observe, physicalLaunch, interpretResult_twice]
  refine ⟨ready_prefix_backend b hadd h buf _ _ ready, ?_⟩
  rw [launch_eq_run b h buf]
  exact sync.all_done

/-- Every actual consumer poll after the stream boundary is ready, regardless
of per-stage program order or interleaving. -/
theorem stream_consumer_poll (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (plan : Plan) (mem : Memory V)
    (done : StageDone h .producers (producerState b h plan mem))
    (cs : List Nat) (p q : Nat) (hp : p < h.numSms)
    (poll : ((Wide.run b h (activeSchedule h .consumers cs) (producerState b h plan mem)).2.pc p).phase = .poll q) :
    let s := (Wide.run b h (activeSchedule h .consumers cs) (producerState b h plan mem)).2
    s.values.mem.locks (q : Int) = 1 ∧ ((Wide.step b h p s).2.pc p).phase = .gather q 0 := by
  apply consumer_poll_immediately_ready b h buf _ p q hp
  · exact run_controlBound h b _ _ (run_controlBound h b _ _ (by intro p; trivial))
  · exact run_dependencyBound b h _ _ (run_dependencyBound b h _ _ (by intro p; trivial))
  · exact every_consumer_prefix_ready b h cs _ (every_completed_producer_stage_ready b h _ mem done)
  · exact poll

inductive Failure where
  | setup | partialMetadata | outputMetadata | lockMetadata | producerSubmission | consumerSubmission
  deriving DecidableEq, Repr
inductive HostEvent where
  | queryPartial | queryOutput | queryLocks | submitProducer | submitConsumer
  deriving DecidableEq, Repr

structure HostResult where
  trace : List HostEvent
  submitted : List Stage
  flags : Option ViewFlags
  returned : Except Failure Unit

/-- Ordinary host API outcomes. Error payloads are abstracted to the source
stage that fails. All three is_neg queries precede either submission. -/
def hostCall (partialFlag outputFlag lockFlag : Except Unit Bool)
    (producerAccepted consumerAccepted : Bool) : HostResult :=
  match partialFlag with
  | .error _ => ⟨[.queryPartial], [], none, .error .partialMetadata⟩
  | .ok p => match outputFlag with
    | .error _ => ⟨[.queryPartial,.queryOutput], [], none, .error .outputMetadata⟩
    | .ok c => match lockFlag with
      | .error _ => ⟨[.queryPartial,.queryOutput,.queryLocks], [], none, .error .lockMetadata⟩
      | .ok l =>
        let flags : ViewFlags := ⟨p,c,l⟩
        if !producerAccepted then
          ⟨[.queryPartial,.queryOutput,.queryLocks,.submitProducer], [], some flags, .error .producerSubmission⟩
        else if !consumerAccepted then
          ⟨[.queryPartial,.queryOutput,.queryLocks,.submitProducer,.submitConsumer], [.producers], some flags, .error .consumerSubmission⟩
        else
          ⟨[.queryPartial,.queryOutput,.queryLocks,.submitProducer,.submitConsumer], [.producers,.consumers], some flags, .ok ()⟩

def setupThenCall (setupSucceeded : Bool) (p c l : Except Unit Bool) (first second : Bool) : HostResult :=
  if setupSucceeded then hostCall p c l first second else ⟨[],[],none,.error .setup⟩

theorem metadata_failure_submits_nothing (p c l : Except Unit Bool) (first second : Bool)
    (failure : p.isOk = false ∨ c.isOk = false ∨ l.isOk = false) :
    (hostCall p c l first second).submitted = [] := by
  cases p <;> cases c <;> cases l <;> simp_all [hostCall, Except.isOk, Except.toBool]

theorem producer_submission_failure (p c l : Bool) (second : Bool) :
    (hostCall (.ok p) (.ok c) (.ok l) false second).submitted = [] ∧
    (hostCall (.ok p) (.ok c) (.ok l) false second).returned = .error .producerSubmission := by
  simp [hostCall]

theorem consumer_submission_failure (p c l : Bool) :
    (hostCall (.ok p) (.ok c) (.ok l) true false).submitted = [.producers] ∧
    (hostCall (.ok p) (.ok c) (.ok l) true false).returned = .error .consumerSubmission := by
  simp [hostCall]

theorem successful_submission_returns_none (p c l : Bool) :
    hostCall (.ok p) (.ok c) (.ok l) true true =
      ⟨[.queryPartial,.queryOutput,.queryLocks,.submitProducer,.submitConsumer],
        [.producers,.consumers],some ⟨p,c,l⟩,.ok ()⟩ := rfl

/-- The accepted submission prefix constrains which stages can have executed.
A rejection of phase two retains all earlier phase-one prefix effects. -/
structure AllowedExecution (b : Backend V) (h : HostArgs) (host : HostResult)
    (plan : Plan) (mem : Memory V) : Prop where
  producerSubmitted : plan.producers ≠ [] → Stage.producers ∈ host.submitted
  consumerSubmitted : plan.consumers ≠ [] → Stage.consumers ∈ host.submitted
  stream : StreamPrefix b h plan mem

inductive DeviceStatus where | pending | synchronized | failed deriving DecidableEq, Repr

def DeviceObservation (b : Backend V) (h : HostArgs) (plan : Plan) (mem : Memory V)
    (status : DeviceStatus) : Prop :=
  match status with
  | .synchronized => Synchronized b h plan mem
  | .pending | .failed => True

/-- Finite-prefix bounds, values, logical views, and all backing memory frames
remain available for pending work and failure. None return alone supplies no
synchronization premise. -/
theorem allowed_prefix_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (encoding : SignEncoding V) (flags : ViewFlags) (h : HostArgs) (buf : Buffers h)
    (host : HostResult) (plan : Plan) (raw : Memory V)
    (_allowed : AllowedExecution b h host plan (interpretMemory encoding flags raw))
    (ready : PublishedReady b h (interpretMemory encoding flags raw)) :
    PrefixFacts b h buf (flatten h plan) (interpretMemory encoding flags raw)
      (interpretResult encoding flags (observe b encoding flags h plan raw)) := by
  simp only [observe, physicalLaunch, interpretResult_twice]
  exact ready_prefix_backend b hadd h buf _ _ ready

theorem no_submissions_no_effect (b : Backend V) (encoding : SignEncoding V)
    (flags : ViewFlags) (h : HostArgs) (buf : Buffers h) (host : HostResult)
    (plan : Plan) (raw : Memory V) (none : host.submitted = [])
    (allowed : AllowedExecution b h host plan (interpretMemory encoding flags raw)) :
    (observe b encoding flags h plan raw).events = [] ∧
      (observe b encoding flags h plan raw).final.values.mem = raw := by
  have hp : plan.producers = [] := by
    by_contra hn
    have hh := allowed.producerSubmitted hn
    simp [none] at hh
  have hc : plan.consumers = [] := by
    by_contra hn
    have hh := allowed.consumerSubmitted hn
    simp [none] at hh
  simp [observe, flatten, activeSchedule, hp, hc, physicalLaunch,
    launch_eq_run b h buf, Wide.run, interpretResult, interpretState, interpretValues,
    Wide.initial, VeriTile.Bench.TritonBenchG.SpinningLock.initial, interpretMemory_twice]

#axiomsClean two_source_invocations
#axiomsClean producer_leader_acc_fresh
#axiomsClean stream_consumer_poll
#axiomsClean synchronized_complete_backend
#axiomsClean allowed_prefix_backend
#axiomsClean no_submissions_no_effect
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Phased

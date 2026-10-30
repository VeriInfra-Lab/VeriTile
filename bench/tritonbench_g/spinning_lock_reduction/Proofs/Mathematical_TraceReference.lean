import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_ConcurrentModel

/-! Independent observation-list specification. It has no program counter and
stores no accumulator: every store payload is recomputed by an ordered fold of
its own prior reads. Events supply no float values or final-state hypotheses.
The full source scheduler produces these addresses and order. -/
namespace VeriTile.Bench.TritonBenchG.SpinningLock

structure Reference (V : Type) where
  mem : Memory V
  reads : Nat → Nat → List V

def Reference.value (b : Backend V) (r : Reference V) (p l : Nat) : V :=
  (r.reads p l).foldl b.add b.zero

def Reference.initial (mem : Memory V) : Reference V := ⟨mem, fun _ _ => []⟩

def observe (b : Backend V) (e : Event) (r : Reference V) : Reference V :=
  match e with
  | .silent | .poll _ => r
  | .add p l a => { r with reads := put r.reads p (put (r.reads p) l (r.reads p l ++ [r.mem.partials a])) }
  | .publish p l a => { r with mem.partials := put r.mem.partials a (r.value b p l) }
  | .release a => { r with mem.locks := put r.mem.locks a 1 }
  | .output p l a => { r with mem.output := put r.mem.output a (r.value b p l) }

def replay (b : Backend V) (events : List Event) (r : Reference V) : Reference V :=
  events.foldl (fun r e => observe b e r) r

structure Related (b : Backend V) (s : State V) (r : Reference V) : Prop where
  memory : s.mem = r.mem
  values : ∀ p l, s.acc p l = r.value b p l

theorem related_initial (b : Backend V) (mem : Memory V) :
    Related b (initial b mem) (Reference.initial mem) := by
  constructor
  · rfl
  · intro p l; rfl

theorem execute_related (b : Backend V) (e : Event) (s : State V) (r : Reference V)
    (h : Related b s r) : Related b (execute b e s) (observe b e r) := by
  rcases h with ⟨hm, hv⟩
  cases e with
  | silent => exact ⟨hm, hv⟩
  | poll a => exact ⟨hm, hv⟩
  | release a =>
    constructor
    · simp only [execute, observe, hm]
    · exact hv
  | publish p l a =>
    constructor
    · simp only [execute, observe, hm, hv]
    · exact hv
  | output p l a =>
    constructor
    · simp only [execute, observe, hm, hv]
    · exact hv
  | add p l a =>
    constructor
    · exact hm
    · intro q j
      by_cases hp : q = p
      · subst q
        by_cases hl : j = l
        · subst j
          simp [execute, observe, Reference.value, put, hm, hv, List.foldl_append]
        · simpa [execute, observe, Reference.value, put, hl] using hv p j
      · simpa [execute, observe, Reference.value, put, hp] using hv q j

theorem step_related (b : Backend V) (c : Config) (p : Nat) (s : State V) (r : Reference V)
    (h : Related b s r) : Related b (step b c p s).2 (observe b (step b c p s).1 r) := by
  have hnext := execute_related b (decode c p s).1 s r h
  exact ⟨hnext.memory, hnext.values⟩

/-- Any finite scheduler, with failed CAS retries retained, refines the separate
observation-list specification. No success, ready-lock, progress, or hstep
hypothesis is required. This theorem does not assert native GPU refinement. -/
theorem run_related (b : Backend V) (c : Config) (schedule : List Nat)
    (s : State V) (r : Reference V) (h : Related b s r) :
    Related b (run b c schedule s).2 (replay b (run b c schedule s).1 r) := by
  induction schedule generalizing s r with
  | nil => exact h
  | cons p ps ih =>
    simpa only [run, replay, List.foldl_cons] using
      ih (step b c p s).2 (observe b (step b c p s).1 r) (step_related b c p s r h)

/-- Full-loop memory and accumulator specification, including prefixes that
block, finish all nine rounds, or fault on undefined signed division. -/
theorem full_schedule_value_memory (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) :
    let result := run b c schedule (initial b mem)
    let reference := replay b result.1 (Reference.initial mem)
    result.2.mem = reference.mem ∧ ∀ p lane,
      result.2.acc p lane = (reference.reads p lane).foldl b.add b.zero := by
  exact ⟨(run_related b c schedule _ _ (related_initial b mem)).memory,
    (run_related b c schedule _ _ (related_initial b mem)).values⟩

/-- The reference's output frame depends only on output events, not on a
postulated final frame or the original memory contents. -/
theorem replay_output_frame (b : Backend V) (events : List Event)
    (r : Reference V) (address : Int)
    (h : ∀ p l, Event.output p l address ∉ events) :
    (replay b events r).mem.output address = r.mem.output address := by
  induction events generalizing r with
  | nil => rfl
  | cons e es ih =>
    rw [replay, List.foldl_cons]
    have ht : ∀ p l, Event.output p l address ∉ es := by
      intro p l he; exact h p l (List.mem_cons_of_mem _ he)
    change (replay b es (observe b e r)).mem.output address = _
    rw [ih _ ht]
    cases e with
    | output p l a =>
      have ha : address ≠ a := by
        intro he; subst a; exact h p l (List.mem_cons_self ..)
      simp [observe, put, ha]
    | _ => rfl

theorem full_schedule_output_frame (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) (address : Int)
    (h : ∀ p l, Event.output p l address ∉ (run b c schedule (initial b mem)).1) :
    (run b c schedule (initial b mem)).2.mem.output address = mem.output address := by
  rw [(run_related b c schedule _ _ (related_initial b mem)).memory]
  exact replay_output_frame b _ _ address h

theorem replay_partials_frame (b : Backend V) (events : List Event)
    (r : Reference V) (address : Int)
    (h : ∀ p l, Event.publish p l address ∉ events) :
    (replay b events r).mem.partials address = r.mem.partials address := by
  induction events generalizing r with
  | nil => rfl
  | cons e es ih =>
    rw [replay, List.foldl_cons]
    have ht : ∀ p l, Event.publish p l address ∉ es := by
      intro p l he; exact h p l (List.mem_cons_of_mem _ he)
    change (replay b es (observe b e r)).mem.partials address = _
    rw [ih _ ht]
    cases e with
    | publish p l a =>
      have ha : address ≠ a := by
        intro he; subst a; exact h p l (List.mem_cons_self ..)
      simp [observe, put, ha]
    | _ => rfl

theorem replay_locks_frame (b : Backend V) (events : List Event)
    (r : Reference V) (address : Int)
    (h : Event.release address ∉ events) :
    (replay b events r).mem.locks address = r.mem.locks address := by
  induction events generalizing r with
  | nil => rfl
  | cons e es ih =>
    rw [replay, List.foldl_cons]
    have ht : Event.release address ∉ es := by
      intro he; exact h (List.mem_cons_of_mem _ he)
    change (replay b es (observe b e r)).mem.locks address = _
    rw [ih _ ht]
    cases e with
    | release a =>
      have ha : address ≠ a := by
        intro he; subst a; exact h (List.mem_cons_self ..)
      simp [observe, put, ha]
    | _ => rfl

theorem full_schedule_frame (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) (address : Int) :
    let result := run b c schedule (initial b mem)
    ((∀ p l, Event.publish p l address ∉ result.1) → result.2.mem.partials address = mem.partials address) ∧
    ((∀ p l, Event.output p l address ∉ result.1) → result.2.mem.output address = mem.output address) ∧
    (Event.release address ∉ result.1 → result.2.mem.locks address = mem.locks address) := by
  dsimp only
  rw [(run_related b c schedule _ _ (related_initial b mem)).memory]
  exact ⟨replay_partials_frame b _ _ address, replay_output_frame b _ _ address,
    replay_locks_frame b _ _ address⟩

#axiomsClean full_schedule_frame
#axiomsClean full_schedule_value_memory
#axiomsClean full_schedule_output_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock

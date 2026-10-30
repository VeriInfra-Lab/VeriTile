import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_SourceBridge

noncomputable section
open Classical

/-! Fresh-lock invariant for arbitrary initial P/C and every finite SC prefix.
Only `zero + zero = zero` is needed. In particular no finite-input hypothesis
is imposed on the unread initial P buffer. -/
namespace VeriTile.Bench.TritonBenchG.SpinningLock

/-- Lane publication is recorded by the actual source PC. A gather PC can only
be reached after a successful poll; publication and ready locks persist. -/
def ControlZeros (b : Backend V) (c : Config) (s : State V) (p : Nat) : Prop :=
  match (s.control p).phase with
  | .publish n => ∀ l, l < n → l < c.lanes → s.mem.partials (c.pAddr (c.pid p) l) = b.zero
  | .release => ∀ l, l < c.lanes → s.mem.partials (c.pAddr (c.pid p) l) = b.zero
  | .gather q _ => ∀ l, l < c.lanes → s.mem.partials (c.pAddr q l) = b.zero
  | _ => True

structure ZeroInvariant (b : Backend V) (c : Config) (s : State V) : Prop where
  acc : ∀ p l, s.acc p l = b.zero
  ready : ∀ q : I32, s.mem.locks q.toInt = 1 →
    ∀ l, l < c.lanes → s.mem.partials (c.pAddr q l) = b.zero
  control : ∀ p, ControlZeros b c s p

/-- Sufficient initial publication state for a later invocation. Fresh locks
establish it without constraining initial P; a previous launch preserves it. -/
def PublishedReady (b : Backend V) (c : Config) (mem : Memory V) : Prop :=
  ∀ q : I32, mem.locks q.toInt = 1 →
    ∀ l, l < c.lanes → mem.partials (c.pAddr q l) = b.zero

theorem initial_readyInvariant (b : Backend V) (c : Config) (mem : Memory V)
    (ready : PublishedReady b c mem) : ZeroInvariant b c (initial b mem) :=
  ⟨fun _ _ => rfl, ready, fun _ => True.intro⟩

theorem fresh_publishedReady (b : Backend V) (c : Config) (mem : Memory V)
    (fresh : ∀ a, mem.locks a = 0) : PublishedReady b c mem := by
  intro q h
  have h01 : (0 : I32) ≠ 1 := by decide
  exact False.elim (h01 ((fresh q.toInt).symm.trans h))

theorem initial_zeroInvariant (b : Backend V) (c : Config) (mem : Memory V)
    (fresh : ∀ a, mem.locks a = 0) : ZeroInvariant b c (initial b mem) :=
  initial_readyInvariant b c mem (fresh_publishedReady b c mem fresh)

theorem execute_preserves_partial_zero (b : Backend V) (e : Event) (s : State V)
    (hz : ∀ p l, s.acc p l = b.zero) (a : Int) (h : s.mem.partials a = b.zero) :
    (execute b e s).mem.partials a = b.zero := by
  cases e <;> simp [execute, put, hz, h]

theorem execute_acc_zero (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (e : Event) (s : State V) (hz : ∀ p l, s.acc p l = b.zero)
    (hr : ∀ p l a, e = .add p l a → s.mem.partials a = b.zero) :
    ∀ p l, (execute b e s).acc p l = b.zero := by
  cases e with
  | add p l a =>
    have ha := hr p l a rfl
    intro q j
    by_cases hq : q = p <;> by_cases hj : j = l <;> simp [execute, put, hz, ha, hadd, hq, hj]
  | _ => exact hz

theorem controlZeros_mono (b : Backend V) (c : Config) (s t : State V) (p : Nat)
    (hc : (t.control p).phase = (s.control p).phase)
    (hm : ∀ a, s.mem.partials a = b.zero → t.mem.partials a = b.zero)
    (h : ControlZeros b c s p) : ControlZeros b c t p := by
  unfold ControlZeros at *
  rw [hc]
  cases hp : (s.control p).phase <;> simp only [hp] at h ⊢
  all_goals first | trivial | (intros; apply hm; apply h <;> assumption)

theorem decode_add_zero (b : Backend V) (c : Config) (s : State V) (p : Nat)
    (h : ZeroInvariant b c s) :
    ∀ q l a, (decode c p s).1 = .add q l a → s.mem.partials a = b.zero := by
  intro q l a he
  have hc := h.control p
  by_cases hg : p ≥ c.grid
  · simp [decode, hg] at he
  · cases hp : (s.control p).phase <;>
      simp only [decode, if_neg hg, hp] at he
    all_goals try split_ifs at he
    all_goals simp_all [ControlZeros]
    rcases he with ⟨rfl, rfl, rfl⟩
    exact hc _ (by assumption)

theorem decode_release_zero (b : Backend V) (c : Config) (s : State V) (p : Nat)
    (h : ZeroInvariant b c s) :
    ∀ a, (decode c p s).1 = .release a → ∀ q : I32, q.toInt = a →
      ∀ l, l < c.lanes → s.mem.partials (c.pAddr q l) = b.zero := by
  intro a he q hq l hl
  have hc := h.control p
  by_cases hg : p ≥ c.grid
  · simp [decode, hg] at he
  · cases hp : (s.control p).phase <;>
      simp only [decode, if_neg hg, hp] at he
    all_goals try split_ifs at he
    all_goals simp_all [ControlZeros]
    have hpid : q = c.pid p := BitVec.toInt_inj.mp (hq.trans he.symm)
    subst q
    exact hc l hl

theorem decode_control_zero (b : Backend V) (c : Config) (s : State V) (p : Nat)
    (h : ZeroInvariant b c s) :
    ControlZeros b c
      { execute b (decode c p s).1 s with control := put s.control p (decode c p s).2 } p := by
  have hc := h.control p
  by_cases hg : p ≥ c.grid
  · simpa [ControlZeros, decode, hg, execute] using hc
  · cases hp : (s.control p).phase with
    | branch =>
      simp only [ControlZeros, put_same, decode, if_neg hg, hp]
      split_ifs <;> simp [execute]
    | scan next =>
      simp only [ControlZeros, put_same, decode, if_neg hg, hp]
      split_ifs <;> trivial
    | poll next =>
      simp only [ControlZeros, put_same, decode, if_neg hg, hp]
      split_ifs with hr
      · simpa [execute] using h.ready next (by simpa using hr)
      · trivial
    | gather next n =>
      simp only [ControlZeros, put_same, decode, if_neg hg, hp]
      split_ifs
      · simpa [ControlZeros, hp, execute] using hc
      · trivial
    | publish n =>
      simp only [ControlZeros, put_same, decode, if_neg hg, hp]
      split_ifs with hn
      · intro l hl hlanes
        by_cases heq : l = n
        · subst l; simp [execute, h.acc]
        · have hlt : l < n := by omega
          have hz : s.mem.partials (c.pAddr (c.pid p) l) = b.zero :=
            (show ∀ l, l < n → l < c.lanes → s.mem.partials (c.pAddr (c.pid p) l) = b.zero by
              simpa [ControlZeros, hp] using hc) l hlt hlanes
          exact execute_preserves_partial_zero b (.publish p n (c.pAddr (c.pid p) n)) s h.acc _ hz
      · intro l hl
        exact (show ∀ l, l < n → l < c.lanes → s.mem.partials (c.pAddr (c.pid p) l) = b.zero by
          simpa [ControlZeros, hp] using hc) l (by omega) hl
    | release => simp [ControlZeros, decode, hg, hp]
    | output n =>
      simp only [ControlZeros, put_same, decode, if_neg hg, hp]
      split_ifs <;> trivial
    | done => simp [ControlZeros, decode, hg, hp]
    | fault => simp [ControlZeros, decode, hg, hp]

theorem step_zeroInvariant (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (c : Config) (s : State V) (p : Nat) (h : ZeroInvariant b c s) :
    ZeroInvariant b c (step b c p s).2 := by
  refine ⟨execute_acc_zero b hadd _ s h.acc (decode_add_zero b c s p h), ?_, ?_⟩
  · intro q hq l hl
    have hr := decode_release_zero b c s p h
    have hp := execute_preserves_partial_zero b (decode c p s).1 s h.acc
    change (execute b (decode c p s).1 s).mem.partials _ = _
    change (execute b (decode c p s).1 s).mem.locks _ = _ at hq
    cases he : (decode c p s).1 with
    | release a =>
      by_cases hqa : q.toInt = a
      · simpa [he, execute] using hr a he q hqa l hl
      · have hold : s.mem.locks q.toInt = 1 := by simpa [he, execute, put, hqa] using hq
        simpa only [he] using hp _ (h.ready q hold l hl)
    | _ =>
      simpa only [he] using hp _ (h.ready q (by simpa [he, execute] using hq) l hl)
  · intro q
    by_cases hqp : q = p
    · subst q; exact decode_control_zero b c s p h
    · apply controlZeros_mono b c s (step b c p s).2 q
      · simp [step, put, hqp]
      · exact execute_preserves_partial_zero b (decode c p s).1 s h.acc
      · exact h.control q

theorem run_zeroInvariant (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (c : Config) (schedule : List Nat) (s : State V) (h : ZeroInvariant b c s) :
    ZeroInvariant b c (run b c schedule s).2 := by
  induction schedule generalizing s with
  | nil => exact h
  | cons p ps ih => exact ih _ (step_zeroInvariant b hadd c s p h)


/-- Observation of an actual source store, independent of its payload. -/
def Visited (events : List Event) (a : Int) : Prop :=
  ∃ p l, Event.output p l a ∈ events

theorem visited_cons (e : Event) (es : List Event) (a : Int) :
    Visited (e :: es) a ↔ (∃ p l, e = .output p l a) ∨ Visited es a := by
  simp only [Visited, List.mem_cons, exists_or, eq_comm]

theorem execute_output_value (b : Backend V) (e : Event) (s : State V)
    (hz : ∀ p l, s.acc p l = b.zero) (a : Int) :
    (execute b e s).mem.output a =
      if (∃ p l, e = .output p l a) then b.zero else s.mem.output a := by
  classical
  cases e <;> simp [execute, put, hz, eq_comm]

/-- The mathematical target is zero on visited cells, with the exact input
frame elsewhere. It is not the false assertion that the entire matrix is filled. -/
theorem run_zero_output (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (c : Config) (schedule : List Nat) (s : State V) (h : ZeroInvariant b c s)
    (a : Int) :
    (run b c schedule s).2.mem.output a =
      if Visited (run b c schedule s).1 a then b.zero else s.mem.output a := by
  classical
  induction schedule generalizing s with
  | nil => simp [run, Visited]
  | cons p ps ih =>
    simp only [run]
    rw [ih _ (step_zeroInvariant b hadd c s p h)]
    change (if Visited (run b c ps (step b c p s).2).1 a then b.zero
      else (execute b (decode c p s).1 s).mem.output a) = _
    rw [execute_output_value b _ s h.acc a]
    simp only [visited_cons]
    change (if Visited (run b c ps (step b c p s).2).1 a then b.zero
      else if ∃ q l, (decode c p s).1 = .output q l a then b.zero else s.mem.output a) = _
    by_cases ht : Visited (run b c ps (step b c p s).2).1 a <;>
      by_cases he : ∃ q l, (decode c p s).1 = .output q l a <;>
      simp_all [step]

#axiomsClean run_zeroInvariant
#axiomsClean run_zero_output
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock

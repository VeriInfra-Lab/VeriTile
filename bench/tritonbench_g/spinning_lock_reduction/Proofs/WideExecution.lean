import bench.tritonbench_g.spinning_lock_reduction.Proofs.WideAddress
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_ZeroInvariant

/-! Natural-coordinate source controller for the wide candidate. The original
memory/FP instruction executor, ordered-read reference, zero-step lemmas and
three complete frame proofs are reused. Only the changed control/address
decoder and its invariant lifting are new. Native i64 address equations are
derived in WideAddress, rather than supplied as domain assumptions.
-/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock

theorem run_related (b : Backend V) (h : HostArgs) (schedule : List Nat)
    (s : State V) (reference : Reference V) (related : Related b s.values reference) :
    Related b (run b h schedule s).2.values (replay b (run b h schedule s).1 reference) := by
  induction schedule generalizing s reference with
  | nil => exact related
  | cons p ps ih =>
    simpa only [run, replay, List.foldl_cons] using
      ih (step b h p s).2 (observe b (step b h p s).1 reference)
        (execute_related b (decode h p s).1 s.values reference related)

/-- Complete memory and ordered-read reference, with no fresh-lock premise. -/
theorem full_schedule_value_memory (b : Backend V) (h : HostArgs)
    (schedule : List Nat) (mem : Memory V) :
    let result := run b h schedule (initial b mem)
    let reference := replay b result.1 (Reference.initial mem)
    result.2.values.mem = reference.mem ∧ ∀ p lane,
      result.2.values.acc p lane = (reference.reads p lane).foldl b.add b.zero :=
  ⟨(run_related b h schedule _ _ (related_initial b mem)).memory,
   (run_related b h schedule _ _ (related_initial b mem)).values⟩

/-- Reuses the original reference's full P/C/lock frame theorems. -/
theorem full_schedule_frame (b : Backend V) (h : HostArgs)
    (schedule : List Nat) (mem : Memory V) (a : Int) :
    let result := run b h schedule (initial b mem)
    ((∀ p l, Event.publish p l a ∉ result.1) → result.2.values.mem.partials a = mem.partials a) ∧
    ((∀ p l, Event.output p l a ∉ result.1) → result.2.values.mem.output a = mem.output a) ∧
    (Event.release a ∉ result.1 → result.2.values.mem.locks a = mem.locks a) := by
  dsimp only
  rw [(run_related b h schedule _ _ (related_initial b mem)).memory]
  exact ⟨replay_partials_frame b _ _ a, replay_output_frame b _ _ a, replay_locks_frame b _ _ a⟩

def ControlZeros (b : Backend V) (h : HostArgs) (s : State V) (p : Nat) : Prop :=
  match (s.pc p).phase with
  | .publish n => ∀ l, l < n → l < lanes h → s.values.mem.partials (pAddress h p l) = b.zero
  | .release => ∀ l, l < lanes h → s.values.mem.partials (pAddress h p l) = b.zero
  | .gather q _ => ∀ l, l < lanes h → s.values.mem.partials (pAddress h q l) = b.zero
  | _ => True

def PublishedReady (b : Backend V) (h : HostArgs) (mem : Memory V) : Prop :=
  ∀ q : Nat, mem.locks (q : Int) = 1 → ∀ l, l < lanes h → mem.partials (pAddress h q l) = b.zero

structure ZeroInvariant (b : Backend V) (h : HostArgs) (s : State V) : Prop where
  acc : ∀ p l, s.values.acc p l = b.zero
  ready : PublishedReady b h s.values.mem
  control : ∀ p, ControlZeros b h s p

theorem fresh_ready (b : Backend V) (h : HostArgs) (mem : Memory V)
    (fresh : ∀ a, mem.locks a = 0) : PublishedReady b h mem := by
  intro q hq
  have h01 : (0 : I32) ≠ 1 := by decide
  exact False.elim (h01 ((fresh q).symm.trans hq))

theorem initial_ready (b : Backend V) (h : HostArgs) (mem : Memory V)
    (ready : PublishedReady b h mem) : ZeroInvariant b h (initial b mem) :=
  ⟨fun _ _ => rfl, ready, fun _ => True.intro⟩

theorem control_zeros_mono (b : Backend V) (h : HostArgs) (s t : State V) (p : Nat)
    (hc : (t.pc p).phase = (s.pc p).phase)
    (hm : ∀ a, s.values.mem.partials a = b.zero → t.values.mem.partials a = b.zero)
    (hz : ControlZeros b h s p) : ControlZeros b h t p := by
  unfold ControlZeros at *
  rw [hc]
  cases hp : (s.pc p).phase <;> simp only [hp] at hz ⊢
  all_goals first | trivial | (intros; apply hm; apply hz <;> assumption)

theorem decode_add_zero (b : Backend V) (h : HostArgs) (s : State V) (p : Nat)
    (hz : ZeroInvariant b h s) :
    ∀ q l a, (decode h p s).1 = .add q l a → s.values.mem.partials a = b.zero := by
  intro q l a he
  have hc := hz.control p
  by_cases hg : p ≥ h.numSms
  · simp [decode, hg] at he
  · cases hp : (s.pc p).phase <;> simp only [decode, if_neg hg, hp] at he
    all_goals try split_ifs at he
    all_goals simp_all [ControlZeros]
    rcases he with ⟨rfl, rfl, rfl⟩
    exact hc _ (by assumption)

theorem decode_release_zero (b : Backend V) (h : HostArgs) (s : State V) (p : Nat)
    (hz : ZeroInvariant b h s) :
    ∀ a, (decode h p s).1 = .release a → ∀ q : Nat, (q : Int) = a →
      ∀ l, l < lanes h → s.values.mem.partials (pAddress h q l) = b.zero := by
  intro a he q hq l hl
  have hc := hz.control p
  by_cases hg : p ≥ h.numSms
  · simp [decode, hg] at he
  · cases hp : (s.pc p).phase <;> simp only [decode, if_neg hg, hp] at he
    all_goals try split_ifs at he
    all_goals simp_all [ControlZeros]
    have hqp : q = p := by omega
    subst q
    exact hc l hl

theorem decode_control_zero (b : Backend V) (h : HostArgs) (s : State V) (p : Nat)
    (hz : ZeroInvariant b h s) : ControlZeros b h (step b h p s).2 p := by
  have hc := hz.control p
  by_cases hg : p ≥ h.numSms
  · simpa [ControlZeros, step, decode, hg, execute] using hc
  · cases hp : (s.pc p).phase with
    | branch =>
      simp only [ControlZeros, step, put_same, decode, if_neg hg, hp]
      split_ifs <;> simp [execute]
    | scan next =>
      simp only [ControlZeros, step, put_same, decode, if_neg hg, hp]
      split_ifs <;> trivial
    | poll next =>
      simp only [ControlZeros, step, put_same, decode, if_neg hg, hp]
      split_ifs with hr
      · simpa [execute] using hz.ready next (by simpa using hr)
      · trivial
    | gather next n =>
      simp only [ControlZeros, step, put_same, decode, if_neg hg, hp]
      split_ifs
      · simpa [ControlZeros, hp, execute] using hc
      · trivial
    | publish n =>
      simp only [ControlZeros, step, put_same, decode, if_neg hg, hp]
      split_ifs with hn
      · intro l hl hlanes
        by_cases heq : l = n
        · subst l; simp [execute, hz.acc]
        · have hlt : l < n := by omega
          have hold : s.values.mem.partials (pAddress h p l) = b.zero :=
            (show ∀ l, l < n → l < lanes h → s.values.mem.partials (pAddress h p l) = b.zero by
              simpa [ControlZeros, hp] using hc) l hlt hlanes
          exact execute_preserves_partial_zero b (.publish p n (pAddress h p n)) s.values hz.acc _ hold
      · intro l hl
        exact (show ∀ l, l < n → l < lanes h → s.values.mem.partials (pAddress h p l) = b.zero by
          simpa [ControlZeros, hp] using hc) l (by omega) hl
    | release => simp [ControlZeros, step, decode, hg, hp]
    | output n =>
      simp only [ControlZeros, step, put_same, decode, if_neg hg, hp]
      split_ifs <;> trivial
    | done => simp [ControlZeros, step, decode, hg, hp]
    | fault => simp [ControlZeros, step, decode, hg, hp]

theorem step_zeroInvariant (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (s : State V) (p : Nat) (hz : ZeroInvariant b h s) :
    ZeroInvariant b h (step b h p s).2 := by
  refine ⟨execute_acc_zero b hadd _ s.values hz.acc (decode_add_zero b h s p hz), ?_, ?_⟩
  · intro q hq l hl
    have hr := decode_release_zero b h s p hz
    have hm := execute_preserves_partial_zero b (decode h p s).1 s.values hz.acc
    change (execute b (decode h p s).1 s.values).mem.partials _ = _
    change (execute b (decode h p s).1 s.values).mem.locks _ = _ at hq
    cases he : (decode h p s).1 with
    | release a =>
      by_cases hqa : (q : Int) = a
      · simpa [he, execute] using hr a he q hqa l hl
      · have hold : s.values.mem.locks (q : Int) = 1 := by simpa [he, execute, put, hqa] using hq
        simpa only [he] using hm _ (hz.ready q hold l hl)
    | _ => simpa only [he] using hm _ (hz.ready q (by simpa [he, execute] using hq) l hl)
  · intro q
    by_cases hqp : q = p
    · subst q; exact decode_control_zero b h s p hz
    · apply control_zeros_mono b h s (step b h p s).2 q
      · simp [step, put, hqp]
      · exact execute_preserves_partial_zero b (decode h p s).1 s.values hz.acc
      · exact hz.control q

theorem run_zeroInvariant (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (schedule : List Nat) (s : State V) (hz : ZeroInvariant b h s) :
    ZeroInvariant b h (run b h schedule s).2 := by
  induction schedule generalizing s with
  | nil => exact hz
  | cons p ps ih => exact ih _ (step_zeroInvariant b hadd h s p hz)

theorem run_zero_output (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (schedule : List Nat) (s : State V) (hz : ZeroInvariant b h s) (a : Int) :
    (run b h schedule s).2.values.mem.output a =
      if Visited (run b h schedule s).1 a then b.zero else s.values.mem.output a := by
  induction schedule generalizing s with
  | nil => simp [run, Visited]
  | cons p ps ih =>
    simp only [run]
    rw [ih _ (step_zeroInvariant b hadd h s p hz)]
    change (if Visited (run b h ps (step b h p s).2).1 a then b.zero
      else (execute b (decode h p s).1 s.values).mem.output a) = _
    rw [execute_output_value b _ s.values hz.acc a]
    simp only [visited_cons]
    change (if Visited (run b h ps (step b h p s).2).1 a then b.zero
      else if ∃ q l, (decode h p s).1 = .output q l a then b.zero else s.values.mem.output a) = _
    by_cases ht : Visited (run b h ps (step b h p s).2).1 a <;>
      by_cases he : ∃ q l, (decode h p s).1 = .output q l a <;> simp_all [step]

#axiomsClean full_schedule_value_memory
#axiomsClean full_schedule_frame
#axiomsClean run_zeroInvariant
#axiomsClean run_zero_output
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide

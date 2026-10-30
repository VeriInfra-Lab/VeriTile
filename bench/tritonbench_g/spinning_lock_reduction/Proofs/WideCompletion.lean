import bench.tritonbench_g.spinning_lock_reduction.Proofs.WideTermination
import bench.tritonbench_g.spinning_lock_reduction.Proofs.WideNative
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Completed

noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock

/-- Before the first store to a particular active lane. This PC invariant is
used contrapositively; completion is never postulated as an output equation. -/
def BeforeOutput (ctl : Control) (lane : Nat) : Prop :=
  ctl.round = 1 ∧ match ctl.phase with
    | .output n => n ≤ lane
    | .done => False
    | _ => True

theorem step_beforeOutput (b : Backend V) (c : HostArgs) (s : State V)
    (p l q : Nat) (hp : p < c.numSms) (hl : l < (lanes c))
    (hm : mask c p l = true) (hs : BeforeOutput (s.pc p) l)
    (he : (step b c q s).1 ≠ Event.output p l (cAddress c p l)) :
    BeforeOutput ((step b c q s).2.pc p) l := by
  by_cases hpq : p = q
  · subst q
    have hg : ¬p ≥ c.numSms := by omega
    rcases hs with ⟨hr, hs⟩
    cases hphase : (s.pc p).phase with
    | branch =>
      simp only [step, decode, if_neg hg, hphase, hr, show ¬1 ≥ 10 by decide, if_false, put_same]
      split_ifs <;> simp [BeforeOutput]
    | scan next =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | poll next =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | gather next n =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | publish n =>
      simp only [step, decode, if_neg hg, hphase, hr, put_same]
      split_ifs <;> simp [BeforeOutput]
    | release => simp [step, decode, hg, hphase, hr, BeforeOutput]
    | output n =>
      simp only [hphase] at hs
      have hn : n < (lanes c) := by omega
      have hnl : n ≠ l := by
        intro hh
        subst n
        exact he (by simp [step, decode, hg, hphase, hl, hm])
      simp only [step, decode, if_neg hg, hphase, if_pos hn, put_same, BeforeOutput, hr]
      exact ⟨True.intro, by omega⟩
    | done => simp [hphase] at hs
    | fault => simp [step, decode, hg, hphase, hr, BeforeOutput]
  · simpa [step, put, hpq] using hs

theorem run_beforeOutput (b : Backend V) (c : HostArgs) (schedule : List Nat)
    (s : State V) (p l : Nat) (hp : p < c.numSms) (hl : l < (lanes c))
    (hm : mask c p l = true) (hs : BeforeOutput (s.pc p) l)
    (he : Event.output p l (cAddress c p l) ∉ (run b c schedule s).1) :
    BeforeOutput ((run b c schedule s).2.pc p) l := by
  induction schedule generalizing s with
  | nil => exact hs
  | cons q qs ih =>
    have hh : (step b c q s).1 ≠ Event.output p l (cAddress c p l) := by
      intro hh; exact he (List.mem_cons.mpr (Or.inl hh.symm))
    have ht : Event.output p l (cAddress c p l) ∉ (run b c qs (step b c q s).2).1 := by
      intro ht; exact he (List.mem_cons_of_mem _ ht)
    exact ih _ (step_beforeOutput b c s p l q hp hl hm hs hh) ht

theorem completed_lane_visited (b : Backend V) (c : HostArgs) (schedule : List Nat)
    (mem : Memory V) (p l : Nat) (hp : p < c.numSms) (hl : l < (lanes c))
    (hm : mask c p l = true)
    (hdone : ((run b c schedule (initial b mem)).2.pc p).phase = .done) :
    Event.output p l (cAddress c p l) ∈ (run b c schedule (initial b mem)).1 := by
  by_contra hn
  have hb := run_beforeOutput b c schedule (initial b mem) p l hp hl hm
    (by simp [BeforeOutput, initial]) hn
  simp [BeforeOutput, hdone] at hb

def outputFootprint (h : HostArgs) (a : Int) : Prop :=
  ∃ p < h.numSms, ∃ l < lanes h, mask h p l = true ∧ cAddress h p l = a

theorem completed_footprint (h : HostArgs) (b : Backend V) (schedule : List Nat)
    (mem : Memory V) (hdone : allDone h (run b h schedule (initial b mem)).2 = true)
    (a : Int) :
    Visited (run b h schedule (initial b mem)).1 a ↔ outputFootprint h a := by
  constructor
  · rintro ⟨p, l, he⟩
    obtain ⟨hp, hl, hm, ha⟩ := run_output_origin h b schedule (initial b mem) p l a he
    exact ⟨p, hp, l, hl, hm, ha⟩
  · rintro ⟨p, hp, l, hl, hm, rfl⟩
    refine ⟨p, l, completed_lane_visited b h schedule mem p l hp hl hm ?_⟩
    have hh := (List.all_eq_true.mp hdone) p (List.mem_range.mpr hp)
    simpa using hh

theorem source_footprint_eq_mathematical (h : HostArgs) (buf : Buffers h) (a : Int) :
    outputFootprint h a ↔ mathematicalFootprint h a := by
  constructor
  · rintro ⟨p, hp, l, hl, hm, ha⟩
    have hl' : l < h.bm * h.bn := hl
    have hrow : l / h.bn < h.bm := (Nat.div_lt_iff_lt_mul buf.tileNPositive).mpr hl'
    have hrem : l % h.bn < h.bn := Nat.mod_lt _ buf.tileNPositive
    have hcol : p * h.bn + l % h.bn < h.numSms * h.bn := by
      have hh := Nat.mul_le_mul_right h.bn (show p + 1 ≤ h.numSms by omega)
      simp only [Nat.add_mul, Nat.one_mul] at hh
      omega
    have hmask : l / h.bn < h.m ∧ p * h.bn + l % h.bn < h.n := by simpa [mask] using hm
    obtain ⟨hrm, hcn⟩ := hmask
    refine ⟨l / h.bn, Nat.lt_min.mpr ⟨hrow, hrm⟩,
      p * h.bn + l % h.bn, Nat.lt_min.mpr ⟨hcol, hcn⟩, ?_⟩
    exact ha
  · rintro ⟨row, hr, col, hc, ha⟩
    obtain ⟨hrb, hrm⟩ := Nat.lt_min.mp hr
    obtain ⟨hcs, hcn⟩ := Nat.lt_min.mp hc
    let p := col / h.bn
    let l := row * h.bn + col % h.bn
    have hp : p < h.numSms := (Nat.div_lt_iff_lt_mul buf.tileNPositive).mpr hcs
    have hrem : col % h.bn < h.bn := Nat.mod_lt _ buf.tileNPositive
    have hl : l < lanes h := by
      have hh := Nat.mul_le_mul_right h.bn (show row + 1 ≤ h.bm by omega)
      simp only [Nat.add_mul, Nat.one_mul] at hh
      change row * h.bn + col % h.bn < h.bm * h.bn
      omega
    have hrow : l / h.bn = row := by
      dsimp [l]
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ buf.tileNPositive,
        Nat.div_eq_of_lt hrem, Nat.zero_add]
    have hcol : p * h.bn + l % h.bn = col := by
      dsimp [p, l]
      simp only [Nat.add_mod, Nat.mul_mod_left, Nat.zero_add, Nat.mod_mod]
      simpa [Nat.mul_comm] using Nat.div_add_mod col h.bn
    refine ⟨p, hp, l, hl, ?_, ?_⟩
    · simpa [mask, hrow, hcol] using And.intro hrm hcn
    · change ((l / h.bn) * h.strideM + (p * h.bn + l % h.bn) * h.strideN : Nat) = a
      rw [hrow, hcol]
      exact ha

structure WrapperResult (V : Type) where
  returned : Unit
  events : List Event
  final : State V

def launch (b : Backend V) (h : HostArgs) (schedule : List Nat) (mem : Memory V) : WrapperResult V :=
  let result := nativeRun b h schedule (initial b mem)
  ⟨(), result.1, result.2⟩

theorem launch_eq_run (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (schedule : List Nat) (mem : Memory V) :
    launch b h schedule mem = ⟨(), (run b h schedule (initial b mem)).1,
      (run b h schedule (initial b mem)).2⟩ := by
  unfold launch
  rw [native_launch_eq b h buf]

/-- Full finite-prefix facts contain exact execution, ordinary allocation bounds,
all three frames, ordered reads, ready-state preservation and written values. -/
structure PrefixFacts (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (schedule : List Nat) (mem : Memory V) (result : WrapperResult V) : Prop where
  execution : result = launch b h schedule mem
  returnedNone : result.returned = ()
  bounded : ∀ e ∈ result.events, EventInBounds buf e
  reference : Related b result.final.values (replay b result.events (Reference.initial mem))
  invariant : ZeroInvariant b h result.final
  output : ∀ a, result.final.values.mem.output a = if Visited result.events a then b.zero else mem.output a
  partialFrame : ∀ a, (∀ p l, Event.publish p l a ∉ result.events) → result.final.values.mem.partials a = mem.partials a
  outputFrame : ∀ a, (∀ p l, Event.output p l a ∉ result.events) → result.final.values.mem.output a = mem.output a
  lockFrame : ∀ a, Event.release a ∉ result.events → result.final.values.mem.locks a = mem.locks a

theorem ready_prefix_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Buffers h) (schedule : List Nat) (mem : Memory V)
    (ready : PublishedReady b h mem) : PrefixFacts b h buf schedule mem (launch b h schedule mem) := by
  have hz := initial_ready b h mem ready
  constructor
  · rfl
  all_goals rw [launch_eq_run b h buf]
  · exact run_in_bounds h buf b schedule (initial b mem) (by intro p; trivial)
  · exact run_related b h schedule (initial b mem) (Reference.initial mem) (related_initial b mem)
  · exact run_zeroInvariant b hadd h schedule _ hz
  · exact run_zero_output b hadd h schedule _ hz
  · intro a; exact (full_schedule_frame b h schedule mem a).1
  · intro a; exact (full_schedule_frame b h schedule mem a).2.1
  · intro a; exact (full_schedule_frame b h schedule mem a).2.2

structure CompleteFacts (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (schedule : List Nat) (mem : Memory V) (result : WrapperResult V) : Prop
    extends PrefixFacts b h buf schedule mem result where
  done : allDone h result.final = true

theorem CompleteFacts.mathematical_output (facts : CompleteFacts b h buf schedule mem result) (a : Int) :
    result.final.values.mem.output a = if mathematicalFootprint h a then b.zero else mem.output a := by
  rw [facts.output]
  have he : result.events = (run b h schedule (initial b mem)).1 := by
    rw [facts.execution, launch_eq_run b h buf]
  have hd : allDone h (run b h schedule (initial b mem)).2 = true := by
    have hh := facts.done
    rw [facts.execution, launch_eq_run b h buf] at hh
    exact hh
  rw [he, completed_footprint h b schedule mem hd a, source_footprint_eq_mathematical h buf a]

theorem ready_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Buffers h) (mem : Memory V) (ready : PublishedReady b h mem) :
    ∃ schedule result, CompleteFacts b h buf schedule mem result := by
  obtain ⟨schedule, hd⟩ := source_schedule_completes b h buf mem
  refine ⟨schedule, launch b h schedule mem, ⟨ready_prefix_backend b hadd h buf schedule mem ready, ?_⟩⟩
  rw [launch_eq_run b h buf]
  exact hd

theorem fresh_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (h : HostArgs) (buf : Buffers h) (mem : Memory V) (fresh : ∀ a, mem.locks a = 0) :
    ∃ schedule result, CompleteFacts b h buf schedule mem result :=
  ready_complete_backend b hadd h buf mem (fresh_ready b h mem fresh)

theorem relaunch_complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (previous : CompleteFacts b oldHost oldBuf oldSchedule oldMem oldResult)
    (h : HostArgs) (buf : Buffers h) (hbm : h.bm = oldHost.bm) (hbn : h.bn = oldHost.bn) :
    ∃ schedule result, CompleteFacts b h buf schedule oldResult.final.values.mem result := by
  apply ready_complete_backend b hadd h buf oldResult.final.values.mem
  intro q hq l hl
  have hold := previous.invariant.ready q hq l
  simpa only [pAddress, lanes, hbm, hbn] using
    hold (by simpa only [lanes, hbm, hbn] using hl)

#axiomsClean completed_footprint
#axiomsClean source_footprint_eq_mathematical
#axiomsClean fresh_complete_backend
#axiomsClean relaunch_complete_backend
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide

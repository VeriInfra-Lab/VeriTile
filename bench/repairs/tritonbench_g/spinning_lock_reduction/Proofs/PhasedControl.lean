import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideHost

/-! Source role guards and stage dependency safety. Completion of the producer
stage is a stream sequencing boundary, not an assumed result equation. These
lemmas apply to every completed producer schedule and every consumer prefix. -/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Phased
open VeriTile.Bench.TritonBenchG.SpinningLock Wide

theorem selected64_exact (h : HostArgs) (buf : Buffers h) (stage : Stage)
    (p : Nat) (hp : p < h.numSms) : selected64 h stage p = selected h stage p := by
  cases stage <;> simp only [selected64, selected, reducer64_exact h buf p hp]

theorem nativeSchedule_eq (h : HostArgs) (buf : Buffers h) (stage : Stage) (xs : List Nat) :
    nativeSchedule h stage xs = activeSchedule h stage xs := by
  have he : (fun p => decide (p < h.numSms) && selected64 h stage p) =
      (fun p => decide (p < h.numSms) && selected h stage p) := by
    funext p
    by_cases hp : p < h.numSms
    · simp only [selected64_exact h buf stage p hp]
    · simp [hp]
  unfold nativeSchedule activeSchedule
  rw [he]

/-- Skipped programs return at the guard and have no memory instruction.
The retained PC/accumulator slots for skipped roles are ghost state: only
selected roles have live local variables in either actual invocation. -/
theorem stageRun_eq (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (stage : Stage) (xs : List Nat) (s : Wide.State V) (hs : ControlBound h s) :
    stageRun b h stage xs s = Wide.run b h (activeSchedule h stage xs) s := by
  unfold stageRun
  rw [nativeSchedule_eq h buf]
  exact native_run_eq b h buf _ s hs

/-- Wide.Control before a producer's first release. It cannot reach output or done
without executing release; no fairness or output-value premise occurs here. -/
def BeforeRelease (ctl : Wide.Control) : Prop :=
  ctl.round = 1 ∧ match ctl.phase with
    | .branch | .publish _ | .release | .fault => True
    | _ => False

theorem step_beforeRelease (b : Backend V) (h : HostArgs) (s : Wide.State V)
    (p q : Nat) (hp : p < h.numSms) (red : reducer h p = false)
    (hs : BeforeRelease (s.pc p)) (he : (Wide.step b h q s).1 ≠ .release (p : Int)) :
    BeforeRelease ((Wide.step b h q s).2.pc p) := by
  by_cases hpq : p = q
  · subst q
    have hg : ¬p ≥ h.numSms := by omega
    rcases hs with ⟨hr, hs⟩
    cases hc : (s.pc p).phase with
    | branch =>
      simp only [Wide.step, Wide.decode, if_neg hg, hc, hr, show ¬1 ≥ 10 by decide, if_false, put_same]
      split_ifs <;> simp_all [BeforeRelease, red]
    | publish lane =>
      simp only [Wide.step, Wide.decode, if_neg hg, hc, put_same]
      split_ifs <;> simp [BeforeRelease, hr]
    | release => exact False.elim (he (by simp [Wide.step, Wide.decode, hg, hc]))
    | fault => simpa [BeforeRelease, Wide.step, Wide.decode, hg, hc] using hr
    | _ => simp [hc] at hs
  · simpa [Wide.step, put, hpq] using hs

theorem run_beforeRelease (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (p : Nat) (hp : p < h.numSms) (red : reducer h p = false)
    (hs : BeforeRelease (s.pc p)) (he : Event.release (p : Int) ∉ (Wide.run b h xs s).1) :
    BeforeRelease ((Wide.run b h xs s).2.pc p) := by
  induction xs generalizing s with
  | nil => exact hs
  | cons q qs ih =>
    have hh : (Wide.step b h q s).1 ≠ .release (p : Int) := by
      intro hh
      exact he (List.mem_cons.mpr (Or.inl hh.symm))
    have ht : Event.release (p : Int) ∉ (Wide.run b h qs (Wide.step b h q s).2).1 := by
      intro ht
      exact he (List.mem_cons_of_mem _ ht)
    exact ih _ (step_beforeRelease b h s p q hp red hs hh) ht

theorem run_release_ready (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (a : Int) (he : Event.release a ∈ (Wide.run b h xs s).1) :
    (Wide.run b h xs s).2.values.mem.locks a = 1 := by
  induction xs generalizing s with
  | nil => simp [Wide.run] at he
  | cons p ps ih =>
    rcases List.mem_cons.mp he with he | he
    · apply run_ready_lock b h ps (Wide.step b h p s).2 a
      change (execute b (Wide.step b h p s).1 s.values).mem.locks a = 1
      change Event.release a = (Wide.step b h p s).1 at he
      rw [← he]
      simp [execute]
    · exact ih _ he

theorem completed_producer_ready (b : Backend V) (h : HostArgs) (xs : List Nat)
    (mem : Memory V) (p : Nat) (hp : p < h.numSms) (red : reducer h p = false)
    (done : ((Wide.run b h xs (Wide.initial b mem)).2.pc p).phase = .done) :
    (Wide.run b h xs (Wide.initial b mem)).2.values.mem.locks (p : Int) = 1 := by
  apply run_release_ready b h xs (Wide.initial b mem)
  by_contra hn
  have hb := run_beforeRelease b h xs (Wide.initial b mem) p hp red
    (by simp [BeforeRelease, Wide.initial]) hn
  simp [BeforeRelease, done] at hb

def StageDone (h : HostArgs) (stage : Stage) (s : Wide.State V) : Prop :=
  ∀ p < h.numSms, selected h stage p = true → (s.pc p).phase = .done

theorem every_completed_producer_stage_ready (b : Backend V) (h : HostArgs)
    (xs : List Nat) (mem : Memory V)
    (done : StageDone h .producers (Wide.run b h (activeSchedule h .producers xs) (Wide.initial b mem)).2) :
    ProducersReady h (Wide.run b h (activeSchedule h .producers xs) (Wide.initial b mem)).2 := by
  intro p hp red
  exact completed_producer_ready b h _ mem p hp red (done p hp (by simp [selected, red]))

/-- A stronger reached-state bound records the source loop's increasing
next_pid and its exact dependency interval. -/
def DependencyBound (h : HostArgs) (s : Wide.State V) : Prop := ∀ p,
  match (s.pc p).phase with
  | .scan q => p < q
  | .poll q | .gather q _ => p < q ∧ continues h p q = true
  | _ => True

theorem step_dependencyBound (b : Backend V) (h : HostArgs) (p : Nat)
    (s : Wide.State V) (hs : DependencyBound h s) : DependencyBound h (Wide.step b h p s).2 := by
  intro q
  by_cases hqp : q = p
  · subst q
    have hc := hs p
    by_cases hg : p ≥ h.numSms
    · simpa [Wide.step, Wide.decode, hg] using hc
    · cases hp : (s.pc p).phase with
      | branch =>
        simp only [Wide.step, Wide.decode, if_neg hg, hp, put_same]
        split_ifs <;> simp
      | scan next =>
        simp only [hp] at hc
        simp only [Wide.step, Wide.decode, if_neg hg, hp, put_same]
        split_ifs with hn
        · exact ⟨hc, hn⟩
        · trivial
      | poll next =>
        simp only [hp] at hc
        simp only [Wide.step, Wide.decode, if_neg hg, hp, put_same]
        split_ifs <;> exact hc
      | gather next lane =>
        simp only [hp] at hc
        simp only [Wide.step, Wide.decode, if_neg hg, hp, put_same]
        split_ifs
        · exact hc
        · omega
      | publish lane =>
        simp only [Wide.step, Wide.decode, if_neg hg, hp, put_same]
        split_ifs <;> trivial
      | release => simp [Wide.step, Wide.decode, hg, hp]
      | output lane =>
        simp only [Wide.step, Wide.decode, if_neg hg, hp, put_same]
        split_ifs <;> trivial
      | done => simp [Wide.step, Wide.decode, hg, hp]
      | fault => simp [Wide.step, Wide.decode, hg, hp]
  · simpa [Wide.step, put, hqp] using hs q

theorem run_dependencyBound (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (hs : DependencyBound h s) : DependencyBound h (Wide.run b h xs s).2 := by
  induction xs generalizing s with
  | nil => exact hs
  | cons p ps ih => exact ih _ (step_dependencyBound b h p s hs)

theorem producer_cannot_poll (h : HostArgs) (s : Wide.State V) (p : Nat)
    (bound : ControlBound h s) (red : selected h .producers p = true) :
    ∀ q, (s.pc p).phase ≠ .poll q := by
  intro q hp
  have hb := bound p
  simp only [Wide.NextBound, hp] at hb
  simp [selected, hb.2] at red

theorem consumer_poll_immediately_ready (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (s : Wide.State V) (p q : Nat) (hp : p < h.numSms) (bound : ControlBound h s)
    (dep : DependencyBound h s) (ready : ProducersReady h s) (poll : (s.pc p).phase = .poll q) :
    s.values.mem.locks (q : Int) = 1 ∧
      ((Wide.step b h p s).2.pc p).phase = .gather q 0 := by
  have hb := bound p
  have hd := dep p
  simp only [Wide.NextBound, poll] at hb
  simp only [poll] at hd
  have red := dependency_is_producer h buf p q hp (by omega) hd.1 hb.2 hd.2
  have lock := ready q hb.1 red
  exact ⟨lock, by simp [Wide.step, Wide.decode, Nat.not_le.mpr hp, poll, lock]⟩

/-- At any later consumer prefix, producer readiness cannot be lost. -/
theorem every_consumer_prefix_ready (b : Backend V) (h : HostArgs) (xs : List Nat)
    (s : Wide.State V) (ready : ProducersReady h s) :
    ProducersReady h (Wide.run b h (activeSchedule h .consumers xs) s).2 :=
  fun p hp red => run_ready_lock b h _ s _ (ready p hp red)

#axiomsClean selected64_exact
#axiomsClean every_completed_producer_stage_ready
#axiomsClean producer_cannot_poll
#axiomsClean consumer_poll_immediately_ready
#axiomsClean every_consumer_prefix_ready
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Phased

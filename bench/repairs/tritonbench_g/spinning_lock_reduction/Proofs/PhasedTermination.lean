import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedHost

/-! Constructive completion of each separately submitted phase. The schedules
contain only the programs selected by that actual source invocation. Producer
work has no poll; consumer dependencies are ready at every prefix by the
same-stream theorem in PhasedHost. Ordinary execution of finite nonblocking
CTAs is the runtime progress boundary, with no producer/consumer co-residency
or inter-CTA fairness requirement. -/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Phased
open VeriTile.Bench.TritonBenchG.SpinningLock Wide

def StageScheduled (b : Backend V) (h : HostArgs) (stage : Stage)
    (s t : Wide.State V) : Prop := ∃ xs, (Wide.run b h (activeSchedule h stage xs) s).2 = t

theorem StageScheduled.refl (b : Backend V) (h : HostArgs) (stage : Stage)
    (s : Wide.State V) : StageScheduled b h stage s s := ⟨[], rfl⟩

theorem StageScheduled.trans (hs : StageScheduled b h stage s t)
    (ht : StageScheduled b h stage t u) : StageScheduled b h stage s u := by
  obtain ⟨xs,hxs⟩ := hs
  obtain ⟨ys,hys⟩ := ht
  refine ⟨xs ++ ys, ?_⟩
  simp only [activeSchedule] at hxs hys
  simp only [activeSchedule, List.filter_append]
  rw [Wide.run_append_state, hxs, hys]

theorem solo_stage (hs : Wide.Solo b h p s t) (hp : p < h.numSms)
    (selected : Phased.selected h stage p = true) : StageScheduled b h stage s t := by
  obtain ⟨n,hn⟩ := hs
  refine ⟨List.replicate n p, ?_⟩
  have he : activeSchedule h stage (List.replicate n p) = List.replicate n p := by
    induction n with
    | zero => rfl
    | succ n ih => simpa [activeSchedule, List.replicate_succ, hp, selected] using congrArg (List.cons p) ih
  rw [he]
  exact hn

theorem complete_producers_prefix (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (mem : Memory V) (n : Nat) (hn : n ≤ h.numSms) :
    ∃ s, StageScheduled b h .producers (Wide.initial b mem) s ∧
      (∀ q < n, reducer h q = false →
        (s.pc q).phase = .done ∧ s.values.mem.locks (q : Int) = 1) ∧
      (∀ q < h.numSms, (n ≤ q ∨ reducer h q = true) →
        s.pc q = ⟨1, .branch⟩) := by
  induction n with
  | zero => exact ⟨Wide.initial b mem, StageScheduled.refl b h .producers _, by simp, by simp [Wide.initial]⟩
  | succ n ih =>
    obtain ⟨s, hs, done, untouched⟩ := ih (by omega)
    by_cases red : reducer h n = true
    · refine ⟨s, hs, ?_, ?_⟩
      · intro q hq hred
        have hqn : q ≠ n := by intro he; subst q; simp [red] at hred
        exact done q (by omega) hred
      · intro q hq he
        exact untouched q hq (he.imp_left (by omega))
    · have redfalse : reducer h n = false := Bool.eq_false_iff.mpr red
      obtain ⟨t, ht, htc, htl⟩ := Wide.solo_initial_complete b h buf n s (by omega)
        (by intro he; exact False.elim (red he)) (untouched n (by omega) (Or.inl (by omega)))
      refine ⟨t, hs.trans (solo_stage ht (by omega) (by simp [selected, redfalse])), ?_, ?_⟩
      · intro q hq hred
        by_cases he : q = n
        · subst q; exact ⟨htc, htl redfalse⟩
        · obtain ⟨hqc, hql⟩ := done q (by omega) hred
          exact ⟨by rw [ht.other_control he]; exact hqc, ht.ready_lock hql⟩
      · intro q hq he
        have hqn : q ≠ n := by
          intro hh; subst q
          rcases he with he | he
          · omega
          · exact red he
        rw [ht.other_control hqn]
        exact untouched q hq (he.imp_left (by omega))

theorem complete_producers (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (mem : Memory V) :
    ∃ s, StageScheduled b h .producers (Wide.initial b mem) s ∧ Wide.ProducersReady h s ∧
      (∀ q < h.numSms, reducer h q = false → (s.pc q).phase = .done) ∧
      (∀ q < h.numSms, reducer h q = true → s.pc q = ⟨1, .branch⟩) := by
  obtain ⟨s, hs, done, untouched⟩ := complete_producers_prefix b h buf mem h.numSms (by omega)
  exact ⟨s, hs, fun q hq hr => (done q hq hr).2,
    fun q hq hr => (done q hq hr).1, fun q hq hr => untouched q hq (Or.inr hr)⟩

theorem complete_consumers_prefix (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (s₀ : Wide.State V) (ready₀ : Wide.ProducersReady h s₀)
    (producers₀ : ∀ q < h.numSms, reducer h q = false → (s₀.pc q).phase = .done)
    (leaders₀ : ∀ q < h.numSms, reducer h q = true → s₀.pc q = ⟨1, .branch⟩)
    (n : Nat) (hn : n ≤ h.numSms) :
    ∃ s, StageScheduled b h .consumers s₀ s ∧ Wide.ProducersReady h s ∧
      (∀ q < h.numSms, (q < n ∨ reducer h q = false) → (s.pc q).phase = .done) ∧
      (∀ q < h.numSms, n ≤ q → reducer h q = true → s.pc q = ⟨1, .branch⟩) := by
  induction n with
  | zero =>
    refine ⟨s₀, StageScheduled.refl b h .consumers _, ready₀, ?_, ?_⟩
    · intro q hq he
      rcases he with he | he
      · omega
      · exact producers₀ q hq he
    · intro q hq _ hr; exact leaders₀ q hq hr
  | succ n ih =>
    obtain ⟨s, hs, ready, done, untouched⟩ := ih (by omega)
    by_cases red : reducer h n = true
    · obtain ⟨t, ht, htc, _⟩ := Wide.solo_initial_complete b h buf n s (by omega)
        (fun _ => ready) (untouched n (by omega) (by omega) red)
      refine ⟨t, hs.trans (solo_stage ht (by omega) red), ht.producersReady ready, ?_, ?_⟩
      · intro q hq he
        by_cases hqn : q = n
        · subst q; exact htc
        · rw [ht.other_control hqn]
          apply done q hq
          exact he.imp_left (by omega)
      · intro q hq hqn hr
        rw [ht.other_control (by omega)]
        exact untouched q hq (by omega) hr
    · refine ⟨s, hs, ready, ?_, ?_⟩
      · intro q hq he
        apply done q hq
        rcases he with he | he
        · by_cases hqn : q = n
          · subst q; exact Or.inr (Bool.eq_false_iff.mpr red)
          · exact Or.inl (by omega)
        · exact Or.inr he
      · intro q hq hqn hr
        exact untouched q hq (by omega) hr

theorem source_phases_complete (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (mem : Memory V) : ∃ plan, Synchronized b h plan mem := by
  obtain ⟨s, ⟨ps,hps⟩, ready, producers, leaders⟩ := complete_producers b h buf mem
  obtain ⟨t, ⟨cs,hcs⟩, _, done, _⟩ := complete_consumers_prefix b h buf s ready producers leaders h.numSms (by omega)
  let plan : Plan := ⟨ps,cs⟩
  refine ⟨plan, ?_, ?_⟩
  · intro p hp red
    have rf : reducer h p = false := by simpa [selected] using red
    change ((Wide.run b h (activeSchedule h .producers ps) (Wide.initial b mem)).2.pc p).phase = .done
    rw [hps]
    exact producers p hp rf
  · intro p hp _
    change ((Wide.run b h (activeSchedule h .producers ps ++ activeSchedule h .consumers cs) (Wide.initial b mem)).2.pc p).phase = .done
    rw [Wide.run_append_state, hps, hcs]
    exact done p hp (Or.inl hp)

theorem complete_backend (b : Backend V) (hadd : b.add b.zero b.zero = b.zero)
    (encoding : SignEncoding V) (flags : ViewFlags) (h : HostArgs) (buf : Buffers h)
    (raw : Memory V) (ready : PublishedReady b h (interpretMemory encoding flags raw)) :
    ∃ plan, Synchronized b h plan (interpretMemory encoding flags raw) ∧
      CompleteFacts b h buf (flatten h plan) (interpretMemory encoding flags raw)
        (interpretResult encoding flags (observe b encoding flags h plan raw)) := by
  obtain ⟨plan,sync⟩ := source_phases_complete b h buf (interpretMemory encoding flags raw)
  exact ⟨plan,sync,synchronized_complete_backend b hadd encoding flags h buf plan raw ready sync⟩

#axiomsClean source_phases_complete
#axiomsClean complete_backend
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Phased

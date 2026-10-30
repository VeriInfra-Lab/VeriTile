import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Fresh

/-! Constructive source-level completion. Nonleaders run to completion before
leaders; every dependency is a nonleader in the same group. The schedule is
finite and consists only of actual source steps. No allDone/value premise,
fairness axiom or native scheduler implementation claim is used. -/
noncomputable section
open Classical
namespace VeriTile.Bench.TritonBenchG.SpinningLock

/-- A finite run of one actual program, allowing zero steps. -/
def Solo (b : Backend V) (c : Config) (p : Nat) (s t : State V) : Prop :=
  ∃ n, (run b c (List.replicate n p) s).2 = t

theorem run_append_state (b : Backend V) (c : Config) (xs ys : List Nat) (s : State V) :
    (run b c (xs ++ ys) s).2 = (run b c ys (run b c xs s).2).2 := by
  induction xs generalizing s with
  | nil => rfl
  | cons p ps ih => exact ih (step b c p s).2

theorem solo_refl (b : Backend V) (c : Config) (p : Nat) (s : State V) : Solo b c p s s :=
  ⟨0, rfl⟩

theorem solo_step (b : Backend V) (c : Config) (p : Nat) (s : State V) :
    Solo b c p s (step b c p s).2 := ⟨1, rfl⟩

theorem Solo.trans (hst : Solo b c p s t) (htu : Solo b c p t u) : Solo b c p s u := by
  obtain ⟨n, hn⟩ := hst
  obtain ⟨m, hm⟩ := htu
  refine ⟨n + m, ?_⟩
  rw [← List.replicate_append_replicate, run_append_state, hn, hm]

theorem step_ready_lock (b : Backend V) (c : Config) (p : Nat) (s : State V)
    (a : Int) (ha : s.mem.locks a = 1) : (step b c p s).2.mem.locks a = 1 := by
  simp only [step]
  generalize decode c p s = e
  rcases e with ⟨e, ctl⟩
  cases e <;> simp [execute, put, ha]

theorem run_ready_lock (b : Backend V) (c : Config) (xs : List Nat) (s : State V)
    (a : Int) (ha : s.mem.locks a = 1) : (run b c xs s).2.mem.locks a = 1 := by
  induction xs generalizing s with
  | nil => exact ha
  | cons p ps ih => exact ih _ (step_ready_lock b c p s a ha)

theorem Solo.ready_lock (hst : Solo b c p s t) (ha : s.mem.locks a = 1) :
    t.mem.locks a = 1 := by
  obtain ⟨n, rfl⟩ := hst
  exact run_ready_lock b c _ s a ha

theorem Solo.other_control (hst : Solo b c p s t) (hq : q ≠ p) :
    t.control q = s.control q := by
  obtain ⟨n, rfl⟩ := hst
  induction n generalizing s with
  | zero => rfl
  | succ n ih =>
    simpa [List.replicate_succ, run, step, put, hq] using ih (s := (step b c p s).2)

theorem solo_output (b : Backend V) (c : Config) (p r l : Nat) (s : State V)
    (hp : p < c.grid) (hl : l ≤ c.lanes)
    (hc : s.control p = ⟨r, .output l⟩) :
    ∃ t, Solo b c p s t ∧ t.control p = ⟨r + 1, .branch⟩ := by
  generalize hn : c.lanes - l = remaining
  induction remaining using Nat.strong_induction_on generalizing l s with
  | h remaining ih =>
    by_cases hlt : l < c.lanes
    · let t := (step b c p s).2
      have ht : t.control p = ⟨r, .output (l + 1)⟩ := by
        simp [t, step, decode, Nat.not_le.mpr hp, hc, hlt]
      obtain ⟨u, hu, huc⟩ := ih (c.lanes - (l + 1)) (by omega) (l + 1) t (by omega) ht rfl
      exact ⟨u, (solo_step b c p s).trans hu, huc⟩
    · refine ⟨(step b c p s).2, solo_step b c p s, ?_⟩
      simp [step, decode, Nat.not_le.mpr hp, hc, hlt]

theorem solo_publish (b : Backend V) (c : Config) (p r l : Nat) (s : State V)
    (hp : p < c.grid) (hl : l ≤ c.lanes)
    (hc : s.control p = ⟨r, .publish l⟩) :
    ∃ t, Solo b c p s t ∧ t.control p = ⟨r, .release⟩ := by
  generalize hn : c.lanes - l = remaining
  induction remaining using Nat.strong_induction_on generalizing l s with
  | h remaining ih =>
    by_cases hlt : l < c.lanes
    · let t := (step b c p s).2
      have ht : t.control p = ⟨r, .publish (l + 1)⟩ := by
        simp [t, step, decode, Nat.not_le.mpr hp, hc, hlt]
      obtain ⟨u, hu, huc⟩ := ih (c.lanes - (l + 1)) (by omega) (l + 1) t (by omega) ht rfl
      exact ⟨u, (solo_step b c p s).trans hu, huc⟩
    · refine ⟨(step b c p s).2, solo_step b c p s, ?_⟩
      simp [step, decode, Nat.not_le.mpr hp, hc, hlt]

theorem solo_gather (b : Backend V) (c : Config) (p r l : Nat) (next : I32) (s : State V)
    (hp : p < c.grid) (hl : l ≤ c.lanes)
    (hc : s.control p = ⟨r, .gather next l⟩) :
    ∃ t, Solo b c p s t ∧ t.control p = ⟨r, .scan (next + 1)⟩ := by
  generalize hn : c.lanes - l = remaining
  induction remaining using Nat.strong_induction_on generalizing l s with
  | h remaining ih =>
    by_cases hlt : l < c.lanes
    · let t := (step b c p s).2
      have ht : t.control p = ⟨r, .gather next (l + 1)⟩ := by
        simp [t, step, decode, Nat.not_le.mpr hp, hc, hlt]
      obtain ⟨u, hu, huc⟩ := ih (c.lanes - (l + 1)) (by omega) (l + 1) t (by omega) ht rfl
      exact ⟨u, (solo_step b c p s).trans hu, huc⟩
    · refine ⟨(step b c p s).2, solo_step b c p s, ?_⟩
      simp [step, decode, Nat.not_le.mpr hp, hc, hlt]

theorem host_division_defined (h : HostArgs) (buf : HostBuffers h) (p : Nat)
    (hp : p < h.numSms) : h.config.divisionDefined p = true := by
  have hn : h.config.numSms ≠ (0#32) := by
    intro he
    have hh := signed32_nat h.numSms buf.smsSigned
    change h.config.numSms.toInt = _ at hh
    rw [he] at hh
    have := buf.smsPositive
    change (0 : Int) = _ at hh
    omega
  have hk : h.config.groupK ≠ (0#32) := by
    intro he
    have hh := signed32_nat h.groupK buf.groupSigned
    change h.config.groupK.toInt = _ at hh
    rw [he] at hh
    have := buf.groupPositive
    change (0 : Int) = _ at hh
    omega
  have hpmin : h.config.pid p ≠ (0x80000000#32) := by
    intro he
    have hh : (h.config.pid p).toInt = (p : Int) :=
      signed32_nat p (Nat.lt_trans hp buf.smsSigned)
    rw [he] at hh
    change (-2147483648 : Int) = (p : Int) at hh
    omega
  simp [Config.divisionDefined, hn, hk, hpmin]

theorem host_reducer_iff (h : HostArgs) (buf : HostBuffers h) (p : Nat)
    (hp : p < h.numSms) : h.config.reducer p = true ↔ p % h.groupK = 0 := by
  have hp31 : p < 2 ^ 31 := Nat.lt_trans hp buf.smsSigned
  have hp32 : p < 4294967296 := by omega
  have hk32 : h.groupK < 4294967296 := by have := buf.groupSigned; omega
  have hm : p % h.groupK < 4294967296 := Nat.lt_of_le_of_lt (Nat.mod_le _ _) hp32
  have hrem : (h.config.pid p).srem h.config.groupK = BitVec.ofNat 32 (p % h.groupK) := by
    apply BitVec.eq_of_toNat_eq
    simp [HostArgs.config, Config.pid, BitVec.srem_eq,
      unsigned32_nat p hp31, unsigned32_nat h.groupK buf.groupSigned,
      Nat.mod_eq_of_lt hp32, Nat.mod_eq_of_lt hk32, Nat.mod_eq_of_lt hm]
  simp only [Config.reducer, hrem, beq_iff_eq]
  constructor
  · intro he
    have hh := congrArg BitVec.toNat he
    have hm : p % h.groupK < 4294967296 := Nat.lt_of_le_of_lt (Nat.mod_le _ _) hp32
    simpa [Nat.mod_eq_of_lt hm] using hh
  · intro he
    simp [he]

theorem host_continues_bounds (h : HostArgs) (buf : HostBuffers h) (p q : Nat)
    (hp : p < h.numSms) (hq : q ≤ h.numSms)
    (hc : h.config.continues p (BitVec.ofNat 32 q) = true) :
    q < h.numSms ∧ q < p + h.groupK := by
  have hq31 : q < 2 ^ 31 := Nat.lt_of_le_of_lt hq buf.smsSigned
  have hsum : p + h.groupK < 4294967296 := by
    have := buf.smsSigned
    have := buf.groupSigned
    omega
  have hbound : (h.config.pid p + h.config.groupK).toInt ≤ (p + h.groupK : Nat) := by
    change (BitVec.ofNat 32 p + BitVec.ofNat 32 h.groupK).toInt ≤ _
    rw [← BitVec.ofNat_add, BitVec.toInt_eq_toNat_cond]
    simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hsum]
    split_ifs <;> omega
  simp only [Config.continues, Bool.and_eq_true, BitVec.slt_iff_toInt_lt] at hc
  have hn : h.config.numSms.toInt = (h.numSms : Int) := signed32_nat _ buf.smsSigned
  rw [signed32_nat _ hq31, hn] at hc
  constructor <;> omega

/-- Every scanned dependency of a leader is a nonleader, even when the signed
upper-bound addition wraps: that can only make the source loop stop earlier. -/
theorem dependency_is_producer (h : HostArgs) (buf : HostBuffers h) (p q : Nat)
    (hp : p < h.numSms) (hq : q ≤ h.numSms) (hpq : p < q)
    (hr : h.config.reducer p = true)
    (hc : h.config.continues p (BitVec.ofNat 32 q) = true) :
    h.config.reducer q = false := by
  obtain ⟨hqs, hqk⟩ := host_continues_bounds h buf p q hp hq hc
  have hpmod := (host_reducer_iff h buf p hp).mp hr
  have hk := buf.groupPositive
  have hdiff : 0 < q - p ∧ q - p < h.groupK := by omega
  have hqeq : q = p + (q - p) := by omega
  have hqm : q % h.groupK ≠ 0 := by
    rw [hqeq, Nat.add_mod, hpmod, Nat.zero_add, Nat.mod_eq_of_lt hdiff.2,
      Nat.mod_eq_of_lt hdiff.2]
    omega
  have hnot : h.config.reducer q ≠ true := by
    intro he
    exact hqm ((host_reducer_iff h buf q hqs).mp he)
  exact Bool.eq_false_iff.mpr hnot

/-- Once the producer phase has run, every nonleader lock is ready. -/
def ProducersReady (h : HostArgs) (s : State V) : Prop :=
  ∀ q < h.numSms, h.config.reducer q = false → s.mem.locks (q : Int) = 1

theorem Solo.producersReady (hst : Solo b h.config p s t) (hr : ProducersReady h s) :
    ProducersReady h t := fun q hq hn => hst.ready_lock (hr q hq hn)

theorem solo_scan (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (p r q : Nat) (s : State V) (hp : p < h.numSms) (hpq : p < q) (hq : q ≤ h.numSms)
    (hr : h.config.reducer p = true) (ready : ProducersReady h s)
    (hctl : s.control p = ⟨r, .scan (BitVec.ofNat 32 q)⟩) :
    ∃ t, Solo b h.config p s t ∧ t.control p = ⟨r, .output 0⟩ := by
  generalize hn : h.numSms - q = remaining
  induction remaining using Nat.strong_induction_on generalizing q s with
  | h remaining ih =>
    by_cases hc : h.config.continues p (BitVec.ofNat 32 q) = true
    · obtain ⟨hqs, _⟩ := host_continues_bounds h buf p q hp hq hc
      have hprod := dependency_is_producer h buf p q hp hq hpq hr hc
      let t := (step b h.config p s).2
      have ht : t.control p = ⟨r, .poll (BitVec.ofNat 32 q)⟩ := by
        simp [t, step, decode, (show ¬p ≥ h.config.grid from Nat.not_le.mpr hp), hctl, hc]
      have hlock : t.mem.locks (BitVec.ofNat 32 q).toInt = 1 := by
        rw [signed32_nat _ (Nat.lt_trans hqs buf.smsSigned)]
        exact (solo_step b h.config p s).ready_lock (ready q hqs hprod)
      let u := (step b h.config p t).2
      have hu : u.control p = ⟨r, .gather (BitVec.ofNat 32 q) 0⟩ := by
        simp [u, step, decode, (show ¬p ≥ h.config.grid from Nat.not_le.mpr hp), ht, hlock]
      obtain ⟨v, hv, hvc⟩ := solo_gather b h.config p r 0 (BitVec.ofNat 32 q) u hp (by omega) hu
      have hsv : Solo b h.config p s v :=
        ((solo_step b h.config p s).trans (solo_step b h.config p t)).trans hv
      have hvc' : v.control p = ⟨r, .scan (BitVec.ofNat 32 (q + 1))⟩ := by
        simpa only [BitVec.ofNat_add] using hvc
      obtain ⟨w, hw, hwc⟩ := ih (h.numSms - (q + 1)) (by omega) (q + 1) v
        (by omega) (by omega) (hsv.producersReady ready) hvc' rfl
      exact ⟨w, hsv.trans hw, hwc⟩
    · refine ⟨(step b h.config p s).2, solo_step b h.config p s, ?_⟩
      simp [step, decode, (show ¬p ≥ h.config.grid from Nat.not_le.mpr hp), hctl, hc]

theorem solo_round (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (p r : Nat) (s : State V) (hp : p < h.numSms) (hr : r < 10)
    (ready : h.config.reducer p = true → ProducersReady h s)
    (hctl : s.control p = ⟨r, .branch⟩) :
    ∃ t, Solo b h.config p s t ∧ t.control p = ⟨r + 1, .branch⟩ ∧
      (h.config.reducer p = false → t.mem.locks (p : Int) = 1) := by
  have hg : ¬p ≥ h.config.grid := Nat.not_le.mpr hp
  have hd := host_division_defined h buf p hp
  let u := (step b h.config p s).2
  by_cases red : h.config.reducer p = true
  · have hu : u.control p = ⟨r, .scan (BitVec.ofNat 32 (p + 1))⟩ := by
      simp [u, step, decode, hg, hctl, Nat.not_le.mpr hr, hd, red,
        Config.pid, BitVec.ofNat_add]
    obtain ⟨v, hv, hvc⟩ := solo_scan b h buf p r (p + 1) u hp (by omega) (by omega)
      red ((solo_step b h.config p s).producersReady (ready red)) hu
    obtain ⟨t, ht, htc⟩ := solo_output b h.config p r 0 v hp (by omega) hvc
    exact ⟨t, ((solo_step b h.config p s).trans hv).trans ht, htc, by simp [red]⟩
  · have redfalse : h.config.reducer p = false := Bool.eq_false_iff.mpr red
    have hu : u.control p = ⟨r, .publish 0⟩ := by
      simp [u, step, decode, hg, hctl, Nat.not_le.mpr hr, hd, red]
    obtain ⟨v, hv, hvc⟩ := solo_publish b h.config p r 0 u hp (by omega) hu
    let w := (step b h.config p v).2
    have hw : w.control p = ⟨r, .output 0⟩ := by
      simp [w, step, decode, hg, hvc]
    have hl : w.mem.locks (p : Int) = 1 := by
      have hpint : (h.config.pid p).toInt = (p : Int) :=
        signed32_nat p (Nat.lt_trans hp buf.smsSigned)
      simp [w, step, decode, hg, hvc, execute, hpint]
    obtain ⟨t, ht, htc⟩ := solo_output b h.config p r 0 w hp (by omega) hw
    exact ⟨t, (((solo_step b h.config p s).trans hv).trans
      (solo_step b h.config p v)).trans ht, htc, fun _ => ht.ready_lock hl⟩

theorem solo_rounds (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (p r : Nat) (s : State V) (hp : p < h.numSms) (hr : r ≤ 10)
    (ready : h.config.reducer p = true → ProducersReady h s)
    (hctl : s.control p = ⟨r, .branch⟩) :
    ∃ t, Solo b h.config p s t ∧ (t.control p).phase = .done := by
  generalize hn : 10 - r = remaining
  induction remaining using Nat.strong_induction_on generalizing r s with
  | h remaining ih =>
    by_cases hlt : r < 10
    · obtain ⟨u, hu, huc, _⟩ := solo_round b h buf p r s hp hlt ready hctl
      obtain ⟨t, ht, htc⟩ := ih (10 - (r + 1)) (by omega) (r + 1) u
        (by omega) (fun he => hu.producersReady (ready he)) huc rfl
      exact ⟨t, hu.trans ht, htc⟩
    · refine ⟨(step b h.config p s).2, solo_step b h.config p s, ?_⟩
      simp [step, decode, (show ¬p ≥ h.config.grid from Nat.not_le.mpr hp), hctl,
        show r ≥ 10 by omega]

theorem solo_initial_complete (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (p : Nat) (s : State V) (hp : p < h.numSms)
    (ready : h.config.reducer p = true → ProducersReady h s)
    (hctl : s.control p = ⟨1, .branch⟩) :
    ∃ t, Solo b h.config p s t ∧ (t.control p).phase = .done ∧
      (h.config.reducer p = false → t.mem.locks (p : Int) = 1) := by
  obtain ⟨u, hu, huc, hul⟩ := solo_round b h buf p 1 s hp (by decide) ready hctl
  obtain ⟨t, ht, htc⟩ := solo_rounds b h buf p 2 u hp (by decide)
    (fun he => hu.producersReady (ready he)) huc
  exact ⟨t, hu.trans ht, htc, fun he => ht.ready_lock (hul he)⟩

/-- A concrete finite source schedule, assembled from solo program runs. -/
def Scheduled (b : Backend V) (c : Config) (s t : State V) : Prop :=
  ∃ schedule, (run b c schedule s).2 = t

theorem Scheduled.refl (b : Backend V) (c : Config) (s : State V) : Scheduled b c s s :=
  ⟨[], rfl⟩

theorem Solo.scheduled (hst : Solo b c p s t) : Scheduled b c s t := by
  obtain ⟨n, hn⟩ := hst
  exact ⟨List.replicate n p, hn⟩

theorem Scheduled.trans (hst : Scheduled b c s t) (htu : Scheduled b c t u) :
    Scheduled b c s u := by
  obtain ⟨xs, hx⟩ := hst
  obtain ⟨ys, hy⟩ := htu
  exact ⟨xs ++ ys, by rw [run_append_state, hx, hy]⟩

/-- Producer phase: increasing pid order, skipping leaders. -/
theorem complete_producers_prefix (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (mem : Memory V) (n : Nat) (hn : n ≤ h.numSms) :
    ∃ s, Scheduled b h.config (initial b mem) s ∧
      (∀ q < n, h.config.reducer q = false →
        (s.control q).phase = .done ∧ s.mem.locks (q : Int) = 1) ∧
      (∀ q < h.numSms, (n ≤ q ∨ h.config.reducer q = true) →
        s.control q = ⟨1, .branch⟩) := by
  induction n with
  | zero => exact ⟨initial b mem, Scheduled.refl b h.config _, by simp, by simp [initial]⟩
  | succ n ih =>
    obtain ⟨s, hs, done, untouched⟩ := ih (by omega)
    by_cases red : h.config.reducer n = true
    · refine ⟨s, hs, ?_, ?_⟩
      · intro q hq hred
        have hqn : q ≠ n := by intro he; subst q; simp [red] at hred
        exact done q (by omega) hred
      · intro q hq he
        exact untouched q hq (he.imp_left (by omega))
    · have redfalse : h.config.reducer n = false := Bool.eq_false_iff.mpr red
      obtain ⟨t, ht, htc, htl⟩ := solo_initial_complete b h buf n s (by omega)
        (by intro he; exact False.elim (red he)) (untouched n (by omega) (Or.inl (by omega)))
      refine ⟨t, hs.trans ht.scheduled, ?_, ?_⟩
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

theorem complete_producers (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (mem : Memory V) :
    ∃ s, Scheduled b h.config (initial b mem) s ∧ ProducersReady h s ∧
      (∀ q < h.numSms, h.config.reducer q = false → (s.control q).phase = .done) ∧
      (∀ q < h.numSms, h.config.reducer q = true → s.control q = ⟨1, .branch⟩) := by
  obtain ⟨s, hs, done, untouched⟩ := complete_producers_prefix b h buf mem h.numSms (by omega)
  exact ⟨s, hs, fun q hq hr => (done q hq hr).2,
    fun q hq hr => (done q hq hr).1, fun q hq hr => untouched q hq (Or.inr hr)⟩

/-- Consumer phase: increasing pid order after every producer completed. -/
theorem complete_consumers_prefix (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (s₀ : State V) (ready₀ : ProducersReady h s₀)
    (producers₀ : ∀ q < h.numSms, h.config.reducer q = false → (s₀.control q).phase = .done)
    (leaders₀ : ∀ q < h.numSms, h.config.reducer q = true → s₀.control q = ⟨1, .branch⟩)
    (n : Nat) (hn : n ≤ h.numSms) :
    ∃ s, Scheduled b h.config s₀ s ∧ ProducersReady h s ∧
      (∀ q < h.numSms, (q < n ∨ h.config.reducer q = false) → (s.control q).phase = .done) ∧
      (∀ q < h.numSms, n ≤ q → h.config.reducer q = true → s.control q = ⟨1, .branch⟩) := by
  induction n with
  | zero =>
    refine ⟨s₀, Scheduled.refl b h.config _, ready₀, ?_, ?_⟩
    · intro q hq he
      rcases he with he | he
      · omega
      · exact producers₀ q hq he
    · intro q hq _ hr; exact leaders₀ q hq hr
  | succ n ih =>
    obtain ⟨s, hs, ready, done, untouched⟩ := ih (by omega)
    by_cases red : h.config.reducer n = true
    · obtain ⟨t, ht, htc, _⟩ := solo_initial_complete b h buf n s (by omega)
        (fun _ => ready) (untouched n (by omega) (by omega) red)
      refine ⟨t, hs.trans ht.scheduled, ht.producersReady ready, ?_, ?_⟩
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

/-- Every HostBuffers configuration admits a complete source schedule. The
proof does not assume fresh locks or any payload values; producers first makes
all required dependencies ready even for arbitrary reused lock contents. -/
theorem source_schedule_completes (b : Backend V) (h : HostArgs) (buf : HostBuffers h)
    (mem : Memory V) :
    ∃ schedule, allDone h.config (run b h.config schedule (initial b mem)).2 = true := by
  obtain ⟨s, hs, ready, producers, leaders⟩ := complete_producers b h buf mem
  obtain ⟨t, ht, _, done, _⟩ := complete_consumers_prefix b h buf s ready producers leaders
    h.numSms (by omega)
  obtain ⟨schedule, hfinal⟩ := hs.trans ht
  refine ⟨schedule, ?_⟩
  unfold allDone
  rw [hfinal]
  apply List.all_eq_true.mpr
  intro p hp
  have hp' : p < h.numSms := List.mem_range.mp hp
  simp [done p hp' (Or.inl hp')]

#axiomsClean source_schedule_completes
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock

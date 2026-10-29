import bench.tritonbench_g.spinning_lock_reduction.Proofs.Host

namespace VeriTile.Bench.TritonBenchG.SpinningLock

def NextBound (h : HostArgs) (phase : Phase) : Prop :=
  match phase with
  | .scan q => 0 ≤ q.toInt ∧ q.toInt ≤ (h.numSms : Int)
  | .poll q | .gather q _ => 0 ≤ q.toInt ∧ q.toInt < (h.numSms : Int)
  | _ => True

def ControlBound (h : HostArgs) (s : State V) : Prop :=
  ∀ p, NextBound h (s.control p).phase

theorem i32_increment (q : I32) (hq : 0 ≤ q.toInt) (hi : q.toInt < 2147483647) :
    (q + 1).toInt = q.toInt + 1 := by
  simp only [BitVec.toInt_add, show (1 : I32).toInt = 1 from rfl]
  apply Int.bmod_eq_of_le <;> omega

theorem step_controlBound (h : HostArgs) (hsm : h.numSms < 2 ^ 31)
    (b : Backend V) (s : State V) (p : Nat) (hs : ControlBound h s) :
    ControlBound h (step b h.config p s).2 := by
  intro q
  by_cases hqp : q = p
  · subst q
    have hc := hs p
    by_cases hg : p ≥ h.config.grid
    · simpa [step, decode, hg, ControlBound] using hc
    · have hp : p < h.numSms := Nat.lt_of_not_ge hg
      have hp31 : p < 2 ^ 31 := Nat.lt_trans hp hsm
      have hpid : (h.config.pid p).toInt = (p : Int) := signed32_nat p hp31
      cases hphase : (s.control p).phase with
      | branch =>
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs <;> simp only [NextBound]
        have hinc : (h.config.pid p + 1).toInt = (p : Int) + 1 := by
          rw [i32_increment _ (by rw [hpid]; omega) (by rw [hpid]; omega), hpid]
        rw [hinc]; omega
      | scan next =>
        simp only [NextBound, hphase] at hc
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs with hn
        · simp only [Config.continues, Bool.and_eq_true, BitVec.slt_iff_toInt_lt] at hn
          have hnum : h.config.numSms.toInt = (h.numSms : Int) := signed32_nat _ hsm
          exact ⟨hc.1, by simpa [hnum] using hn.2⟩
        · trivial
      | poll next =>
        simp only [NextBound, hphase] at hc
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs <;> exact hc
      | gather next lane =>
        simp only [NextBound, hphase] at hc
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs
        · exact hc
        · simp only [NextBound]
          rw [i32_increment next hc.1 (by omega)]
          omega
      | publish lane =>
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs <;> trivial
      | release => simp [step, decode, hg, hphase, NextBound]
      | output lane =>
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs <;> trivial
      | done => simp [step, decode, hg, hphase, NextBound]
      | fault => simp [step, decode, hg, hphase, NextBound]
  · simpa [step, put, hqp] using hs q

def EventInBounds (buffers : HostBuffers h) : Event → Prop
  | .silent => True
  | .add _ _ a | .publish _ _ a => 0 ≤ a ∧ a < (buffers.pElements : Int)
  | .poll a | .release a => 0 ≤ a ∧ a < (buffers.lockElements : Int)
  | .output _ _ a => 0 ≤ a ∧ a < (buffers.cElements : Int)

theorem signed_pid_of_bound (h : HostArgs) (hsm : h.numSms < 2 ^ 31)
    (q : I32) (hq : 0 ≤ q.toInt ∧ q.toInt < (h.numSms : Int)) :
    q = BitVec.ofNat 32 q.toInt.toNat ∧ q.toInt.toNat < h.numSms := by
  have hn : q.toInt.toNat < h.numSms := by omega
  constructor
  · apply BitVec.toInt_inj.mp
    rw [signed32_nat _ (Nat.lt_trans hn hsm), Int.toNat_of_nonneg hq.1]
  · exact hn

theorem decode_in_bounds (h : HostArgs) (buffers : HostBuffers h)
    (s : State V) (p : Nat) (hs : ControlBound h s) :
    EventInBounds buffers (decode h.config p s).1 := by
  have hc := hs p
  by_cases hg : p ≥ h.config.grid
  · simp [decode, hg, EventInBounds]
  · have hp : p < h.numSms := Nat.lt_of_not_ge hg
    have hpid : (h.config.pid p).toInt = (p : Int) :=
      signed32_nat p (Nat.lt_trans hp buffers.smsSigned)
    cases hphase : (s.control p).phase with
    | branch =>
      simp only [decode, if_neg hg, hphase]
      split_ifs <;> trivial
    | scan next => simp [decode, hg, hphase, EventInBounds]
    | poll next =>
      simp only [NextBound, hphase] at hc
      simp only [decode, if_neg hg, hphase, EventInBounds]
      have cap := buffers.lockCapacity
      exact ⟨hc.1, by omega⟩
    | gather next lane =>
      simp only [NextBound, hphase] at hc
      simp only [decode, if_neg hg, hphase]
      split_ifs with hl
      · obtain ⟨hq, hn⟩ := signed_pid_of_bound h buffers.smsSigned next hc
        change 0 ≤ h.config.pAddr next lane ∧ _
        rw [hq]
        exact partial_offset_in_allocation h buffers _ _ hn hl
      · trivial
    | publish lane =>
      simp only [decode, if_neg hg, hphase]
      split_ifs with hl
      · exact partial_offset_in_allocation h buffers p lane hp hl
      · trivial
    | release =>
      simp only [decode, if_neg hg, hphase, EventInBounds, hpid]
      exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (Nat.lt_of_lt_of_le hp buffers.lockCapacity)⟩
    | output lane =>
      simp only [decode, if_neg hg, hphase]
      split_ifs with hl hm
      · exact output_offset_in_allocation h buffers p lane hp hl hm
      all_goals trivial
    | done => simp [decode, hg, hphase, EventInBounds]
    | fault => simp [decode, hg, hphase, EventInBounds]

theorem run_in_bounds (h : HostArgs) (buffers : HostBuffers h)
    (b : Backend V) (schedule : List Nat) (s : State V) (hs : ControlBound h s) :
    ∀ e ∈ (run b h.config schedule s).1, EventInBounds buffers e := by
  induction schedule generalizing s with
  | nil => simp [run]
  | cons p ps ih =>
    intro e he
    rcases List.mem_cons.mp he with he | he
    · subst e; exact decode_in_bounds h buffers s p hs
    · exact ih _ (step_controlBound h buffers.smsSigned b s p hs) e he

theorem decode_output_origin (c : Config) (s : State V) (p q l : Nat) (a : Int)
    (he : (decode c p s).1 = .output q l a) :
    q < c.grid ∧ l < c.lanes ∧ c.outputMask q l = true ∧ c.cAddr q l = a := by
  by_cases hg : p ≥ c.grid
  · simp [decode, hg] at he
  · cases hp : (s.control p).phase <;>
      simp only [decode, if_neg hg, hp] at he
    all_goals try split_ifs at he
    all_goals simp_all
    rcases he with ⟨rfl, rfl, rfl⟩
    rfl

theorem run_output_origin (c : Config) (b : Backend V) (schedule : List Nat)
    (s : State V) (q l : Nat) (a : Int)
    (he : Event.output q l a ∈ (run b c schedule s).1) :
    q < c.grid ∧ l < c.lanes ∧ c.outputMask q l = true ∧ c.cAddr q l = a := by
  induction schedule generalizing s with
  | nil => simp [run] at he
  | cons p ps ih =>
    rcases List.mem_cons.mp he with he | he
    · exact decode_output_origin c s p q l a he.symm
    · exact ih _ he

#axiomsClean run_in_bounds
#axiomsClean run_output_origin
end VeriTile.Bench.TritonBenchG.SpinningLock

import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.WideExecution

/-! All reached address/control bounds and exact native i64/u64 scalar expressions.
No native-safe equality is assumed in Buffers. -/
namespace VeriTile.Bench.Repairs.SpinningLock.Wide
open VeriTile.Bench.TritonBenchG.SpinningLock

theorem reducer64_exact (h : HostArgs) (buf : Buffers h) (p : Nat) (hp : p < h.numSms) :
    reducer64 h p = reducer h p := by
  have hp64 : p < 2 ^ 64 := by have := buf.smsGridLimit; omega
  have hm64 : p % h.groupK < 2 ^ 64 := Nat.lt_of_le_of_lt (Nat.mod_le _ _) hp64
  have he : (BitVec.ofNat 64 p).umod (BitVec.ofNat 64 h.groupK) = BitVec.ofNat 64 (p % h.groupK) := by
    apply BitVec.eq_of_toNat_eq
    change (p % 2 ^ 64) % (h.groupK % 2 ^ 64) = (p % h.groupK) % 2 ^ 64
    rw [Nat.mod_eq_of_lt hp64, Nat.mod_eq_of_lt buf.groupScalarLimit, Nat.mod_eq_of_lt hm64]
  simp only [reducer64, he, reducer]
  apply Bool.eq_iff_iff.mpr
  simp only [beq_iff_eq, decide_eq_true_eq]
  constructor
  · intro hh
    have hn := congrArg BitVec.toNat hh
    simpa only [BitVec.toNat_ofNat, BitVec.toNat_zero, Nat.mod_eq_of_lt hm64] using hn
  · intro hh; simp [hh]

theorem leader_upper_bound (h : HostArgs) (buf : Buffers h) (p : Nat)
    (hp : p < h.numSms) (hr : reducer h p = true) : p + h.groupK < 2 ^ 64 := by
  have hmod : p % h.groupK = 0 := by simpa [reducer] using hr
  by_cases hk : h.numSms ≤ h.groupK
  · have hp0 : p = 0 := by rw [Nat.mod_eq_of_lt (by omega)] at hmod; exact hmod
    simpa [hp0] using buf.groupScalarLimit
  · have := buf.smsGridLimit
    omega

theorem continues64_exact (h : HostArgs) (buf : Buffers h) (p q : Nat)
    (hp : p < h.numSms) (hq : q ≤ h.numSms) (hr : reducer h p = true) :
    continues64 h p q = continues h p q := by
  have hq63 : q < 2 ^ 63 := by have := buf.smsGridLimit; omega
  have hn63 : h.numSms < 2 ^ 63 := by have := buf.smsGridLimit; omega
  have hsum := leader_upper_bound h buf p hp hr
  apply Bool.eq_iff_iff.mpr
  simp only [continues64, continues, Bool.and_eq_true, BitVec.ult_iff_toNat_lt,
    BitVec.slt_iff_toInt_lt, decide_eq_true_eq, ← BitVec.ofNat_add]
  rw [signed64_nat _ hq63, signed64_nat _ hn63]
  simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hsum,
    Nat.mod_eq_of_lt (show q < 2 ^ 64 by omega)]
  omega

theorem next64_exact (h : HostArgs) (buf : Buffers h) (q : Nat) (hq : q < h.numSms) :
    (BitVec.ofNat 64 q + 1).toNat = q + 1 := by
  have hb : q + 1 < 2 ^ 64 := by have := buf.smsGridLimit; omega
  rw [show (1 : I64) = BitVec.ofNat 64 1 from rfl, ← BitVec.ofNat_add]
  simp only [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hb]

def NextBound (h : HostArgs) (p : Nat) (phase : Phase) : Prop :=
  match phase with
  | .scan q => q ≤ h.numSms ∧ reducer h p = true
  | .poll q | .gather q _ => q < h.numSms ∧ reducer h p = true
  | _ => True

def ControlBound (h : HostArgs) (s : State V) : Prop := ∀ p, NextBound h p (s.pc p).phase

theorem step_controlBound (h : HostArgs) (b : Backend V) (s : State V) (p : Nat)
    (hs : ControlBound h s) : ControlBound h (step b h p s).2 := by
  intro q
  by_cases hqp : q = p
  · subst q
    have hc := hs p
    by_cases hg : p ≥ h.numSms
    · simpa [step, decode, hg] using hc
    · cases hphase : (s.pc p).phase with
      | branch =>
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs <;> simp_all [NextBound]
        omega
      | scan next =>
        simp only [NextBound, hphase] at hc
        simp only [step, decode, if_neg hg, hphase, put_same]
        split_ifs with hn
        · exact ⟨(show next < p + h.groupK ∧ next < h.numSms by simpa [continues] using hn).2, hc.2⟩
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
        · exact ⟨by omega, hc.2⟩
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

def EventInBounds (buf : Buffers h) : Event → Prop
  | .silent => True
  | .add _ _ a | .publish _ _ a => 0 ≤ a ∧ a < (buf.pElements : Int)
  | .poll a | .release a => 0 ≤ a ∧ a < (buf.lockElements : Int)
  | .output _ _ a => 0 ≤ a ∧ a < (buf.cElements : Int)

theorem decode_in_bounds (h : HostArgs) (buf : Buffers h) (s : State V) (p : Nat)
    (hs : ControlBound h s) : EventInBounds buf (decode h p s).1 := by
  have hc := hs p
  by_cases hg : p ≥ h.numSms
  · simp [decode, hg, EventInBounds]
  · have hp : p < h.numSms := by omega
    cases hphase : (s.pc p).phase with
    | branch => simp only [decode, if_neg hg, hphase]; split_ifs <;> trivial
    | scan next => simp [decode, hg, hphase, EventInBounds]
    | poll next =>
      simp only [NextBound, hphase] at hc
      simp only [decode, if_neg hg, hphase, EventInBounds]
      exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (Nat.lt_of_lt_of_le hc.1 buf.lockCapacity)⟩
    | gather next lane =>
      simp only [NextBound, hphase] at hc
      simp only [decode, if_neg hg, hphase]
      split_ifs with hl
      · exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (partial_nat_bounds h buf next lane hc.1 hl)⟩
      · trivial
    | publish lane =>
      simp only [decode, if_neg hg, hphase]
      split_ifs with hl
      · exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (partial_nat_bounds h buf p lane hp hl)⟩
      · trivial
    | release =>
      simp only [decode, if_neg hg, hphase, EventInBounds]
      exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (Nat.lt_of_lt_of_le hp buf.lockCapacity)⟩
    | output lane =>
      simp only [decode, if_neg hg, hphase]
      split_ifs with hl hm
      · have hm' : lane / h.bn < h.m ∧ p * h.bn + lane % h.bn < h.n := by simpa [mask] using hm
        exact ⟨Int.natCast_nonneg _, Int.ofNat_lt.mpr (buf.outputCapacity _ hm'.1 _ hm'.2)⟩
      all_goals trivial
    | done => simp [decode, hg, hphase, EventInBounds]
    | fault => simp [decode, hg, hphase, EventInBounds]

theorem run_in_bounds (h : HostArgs) (buf : Buffers h) (b : Backend V)
    (schedule : List Nat) (s : State V) (hs : ControlBound h s) :
    ∀ e ∈ (run b h schedule s).1, EventInBounds buf e := by
  induction schedule generalizing s with
  | nil => simp [run]
  | cons p ps ih =>
    intro e he
    rcases List.mem_cons.mp he with he | he
    · subst e; exact decode_in_bounds h buf s p hs
    · exact ih _ (step_controlBound h b s p hs) e he

theorem run_controlBound (h : HostArgs) (b : Backend V) (schedule : List Nat)
    (s : State V) (hs : ControlBound h s) : ControlBound h (run b h schedule s).2 := by
  induction schedule generalizing s with
  | nil => exact hs
  | cons p ps ih => exact ih _ (step_controlBound h b s p hs)

theorem decode_output_origin (h : HostArgs) (s : State V) (p q l : Nat) (a : Int)
    (he : (decode h p s).1 = .output q l a) :
    q < h.numSms ∧ l < lanes h ∧ mask h q l = true ∧ cAddress h q l = a := by
  by_cases hg : p ≥ h.numSms
  · simp [decode, hg] at he
  · cases hp : (s.pc p).phase <;> simp only [decode, if_neg hg, hp] at he
    all_goals try split_ifs at he
    all_goals simp_all
    rcases he with ⟨rfl, rfl, rfl⟩
    rfl

theorem run_output_origin (h : HostArgs) (b : Backend V) (schedule : List Nat)
    (s : State V) (q l : Nat) (a : Int) (he : Event.output q l a ∈ (run b h schedule s).1) :
    q < h.numSms ∧ l < lanes h ∧ mask h q l = true ∧ cAddress h q l = a := by
  induction schedule generalizing s with
  | nil => simp [run] at he
  | cons p ps ih =>
    rcases List.mem_cons.mp he with he | he
    · exact decode_output_origin h s p q l a he.symm
    · exact ih _ he

#axiomsClean reducer64_exact
#axiomsClean continues64_exact
#axiomsClean run_in_bounds
#axiomsClean run_controlBound
#axiomsClean run_output_origin
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Wide

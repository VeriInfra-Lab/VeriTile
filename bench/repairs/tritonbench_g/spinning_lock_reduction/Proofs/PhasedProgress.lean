import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.PhasedHost

/-! A natural-number variant strictly decreases on each nonfinished program
step once polls are ready. This excludes any source-level infinite resident
CTA loop; the runtime need only execute finite, nonblocking CTAs. -/
noncomputable section
open Classical
namespace VeriTile.Bench.Repairs.SpinningLock.Phased
open VeriTile.Bench.TritonBenchG.SpinningLock Wide

def WorkControl (h : HostArgs) (ctl : Wide.Control) : Prop :=
  ctl.round ≤ 10 ∧ match ctl.phase with
    | .branch | .done => True
    | .fault => False
    | .publish l | .output l | .gather _ l => ctl.round < 10 ∧ l ≤ lanes h
    | _ => ctl.round < 10

def WorkBound (h : HostArgs) (s : Wide.State V) : Prop := ∀ p, WorkControl h (s.pc p)

theorem step_workBound (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (p : Nat) (s : Wide.State V) (hs : WorkBound h s) : WorkBound h (Wide.step b h p s).2 := by
  intro q
  by_cases hqp : q = p
  · subst q
    have hc := hs p
    have hk : h.groupK ≠ 0 := by have := buf.groupPositive; omega
    by_cases hg : p ≥ h.numSms
    · simpa [Wide.step, Wide.decode, hg] using hc
    · cases hp : (s.pc p).phase <;>
        simp only [WorkControl, hp] at hc <;>
        simp only [Wide.step, Wide.decode, if_neg hg, hp, put_same]
      all_goals try split_ifs
      all_goals simp_all [WorkControl]
      all_goals omega
  · simpa [Wide.step, put, hqp] using hs q

theorem run_workBound (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (xs : List Nat) (s : Wide.State V) (hs : WorkBound h s) : WorkBound h (Wide.run b h xs s).2 := by
  induction xs generalizing s with
  | nil => exact hs
  | cons p ps ih => exact ih _ (step_workBound b h buf p s hs)

def roundBudget (h : HostArgs) : Nat := (h.numSms + 1) * (lanes h + 3) + 2 * lanes h + 4

def phaseWeight (h : HostArgs) : Wide.Phase → Nat
  | .branch => roundBudget h
  | .scan q => (h.numSms - q) * (lanes h + 3) + lanes h + 3
  | .poll q => (h.numSms - q) * (lanes h + 3) + lanes h + 2
  | .gather q l => (h.numSms - q) * (lanes h + 3) + (lanes h - l) + 1
  | .publish l => (lanes h - l) + lanes h + 3
  | .release => lanes h + 2
  | .output l => (lanes h - l) + 1
  | .done | .fault => 0

def localRank (h : HostArgs) (ctl : Wide.Control) : Nat :=
  if ctl.phase = .done then 0 else (10 - ctl.round) * roundBudget h + phaseWeight h ctl.phase

theorem scan_weight_lt_budget (h : HostArgs) (q : Nat) :
    phaseWeight h (.scan q) < roundBudget h := by
  have hh := Nat.mul_le_mul_right (lanes h + 3) (Nat.sub_le h.numSms q)
  simp only [phaseWeight, roundBudget, Nat.add_mul, Nat.one_mul]
  omega

theorem step_rank_decreases (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (p : Nat) (s : Wide.State V) (hp : p < h.numSms)
    (work : WorkBound h s) (bound : ControlBound h s)
    (pollReady : ∀ q, (s.pc p).phase = .poll q → s.values.mem.locks (q : Int) = 1)
    (notDone : (s.pc p).phase ≠ .done) :
    localRank h ((Wide.step b h p s).2.pc p) < localRank h (s.pc p) := by
  have hw := work p
  have hb := bound p
  have hg : ¬p ≥ h.numSms := by omega
  have hk : h.groupK ≠ 0 := by have := buf.groupPositive; omega
  have budgetpos : 0 < roundBudget h := by unfold roundBudget; omega
  cases hc : (s.pc p).phase with
  | branch =>
    have hr : (s.pc p).round ≤ 10 := hw.1
    simp only [Wide.step, Wide.decode, if_neg hg, hc, put_same, if_neg hk]
    split_ifs with hd red
    · simp [localRank, phaseWeight, hc]
      omega
    · have small := scan_weight_lt_budget h (p + 1)
      simpa [localRank, phaseWeight, hc] using Nat.add_lt_add_left small ((10 - (s.pc p).round) * roundBudget h)
    · have small : phaseWeight h (.publish 0) < roundBudget h := by
        simp [phaseWeight, roundBudget]
        omega
      simpa [localRank, phaseWeight, hc] using Nat.add_lt_add_left small ((10 - (s.pc p).round) * roundBudget h)
  | scan q =>
    simp only [Wide.step, Wide.decode, if_neg hg, hc, put_same]
    split_ifs <;> simp [localRank, phaseWeight, hc] <;> omega
  | poll q =>
    have ready := pollReady q hc
    simp [Wide.step, Wide.decode, hg, hc, ready, localRank, phaseWeight]
  | gather q l =>
    simp only [WorkControl, hc] at hw
    simp only [Wide.NextBound, hc] at hb
    have hq : q < h.numSms := hb.1
    have hl : l ≤ lanes h := hw.2.2
    simp only [Wide.step, Wide.decode, if_neg hg, hc, put_same]
    split_ifs with hlt
    · simp [localRank, phaseWeight, hc]
      omega
    · have heq : l = lanes h := by omega
      have hsub : h.numSms - q = (h.numSms - (q + 1)) + 1 := by omega
      simp [localRank, phaseWeight, hc, heq, hsub, Nat.add_mul]
      omega
  | publish l =>
    simp only [WorkControl, hc] at hw
    have hl : l ≤ lanes h := hw.2.2
    simp only [Wide.step, Wide.decode, if_neg hg, hc, put_same]
    split_ifs with hlt
    · simp [localRank, phaseWeight, hc]
      omega
    · have heq : l = lanes h := by omega
      simp [localRank, phaseWeight, hc, heq]
  | release => simp [Wide.step, Wide.decode, hg, hc, localRank, phaseWeight]
  | output l =>
    simp only [WorkControl, hc] at hw
    have hr : (s.pc p).round < 10 := hw.2.1
    have hl : l ≤ lanes h := hw.2.2
    by_cases hlt : l < lanes h
    · simp [Wide.step, Wide.decode, hg, hc, hlt, localRank, phaseWeight] <;> omega
    · have heq : l = lanes h := by omega
      have hsub : 10 - (s.pc p).round = (10 - ((s.pc p).round + 1)) + 1 := by omega
      simp [Wide.step, Wide.decode, hg, hc, hlt, localRank, phaseWeight, heq, hsub, Nat.add_mul] <;> omega
  | done => exact False.elim (notDone hc)
  | fault => simp [WorkControl, hc] at hw

theorem producer_step_rank (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (xs : List Nat) (mem : Memory V) (p : Nat) (hp : p < h.numSms)
    (active : selected h .producers p = true)
    (notDone : ((Wide.run b h (activeSchedule h .producers xs) (Wide.initial b mem)).2.pc p).phase ≠ .done) :
    let s := (Wide.run b h (activeSchedule h .producers xs) (Wide.initial b mem)).2
    localRank h ((Wide.step b h p s).2.pc p) < localRank h (s.pc p) := by
  have bound := run_controlBound h b (activeSchedule h .producers xs) (Wide.initial b mem) (by intro p; trivial)
  apply step_rank_decreases b h buf p _ hp
  · exact run_workBound b h buf _ _ (by intro p; simp [WorkControl, Wide.initial])
  · exact bound
  · intro q poll
    exact False.elim ((producer_cannot_poll h _ p bound active q) poll)
  · exact notDone

theorem consumer_step_rank (b : Backend V) (h : HostArgs) (buf : Buffers h)
    (plan : Plan) (mem : Memory V)
    (done : StageDone h .producers (producerState b h plan mem))
    (cs : List Nat) (p : Nat) (hp : p < h.numSms)
    (notDone : ((Wide.run b h (activeSchedule h .consumers cs) (producerState b h plan mem)).2.pc p).phase ≠ .done) :
    let s := (Wide.run b h (activeSchedule h .consumers cs) (producerState b h plan mem)).2
    localRank h ((Wide.step b h p s).2.pc p) < localRank h (s.pc p) := by
  apply step_rank_decreases b h buf p _ hp
  · exact run_workBound b h buf _ _ (run_workBound b h buf _ _ (by intro p; simp [WorkControl, Wide.initial]))
  · exact run_controlBound h b _ _ (run_controlBound h b _ _ (by intro p; trivial))
  · intro q poll
    exact (stream_consumer_poll b h buf plan mem done cs p q hp poll).1
  · exact notDone

#axiomsClean step_workBound
#axiomsClean step_rank_decreases
#axiomsClean producer_step_rank
#axiomsClean consumer_step_rank
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.Phased

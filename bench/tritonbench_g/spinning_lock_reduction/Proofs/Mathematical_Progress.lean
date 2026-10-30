import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_Software
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.NormNum
import Mathlib.Logic.Relation

namespace VeriTile.Bench.TritonBenchG.SpinningLock

/-- The source CAS compares and replaces with the same word. A failed poll
cannot make its lock ready and does not change any numerical state. -/
theorem failed_poll_step (b : Backend V) (c : Config) (p : Nat) (next : I32) (s : State V)
    (hp : p < c.grid) (hphase : (s.control p).phase = .poll next)
    (hlock : s.mem.locks next.toInt ≠ 1) :
    step b c p s = (.poll next.toInt, s) := by
  have hctl : { s.control p with phase := .poll next } = s.control p := by
    cases he : s.control p
    simp_all
  have hput : put s.control p (s.control p) = s.control := by
    funext q; by_cases h : q = p <;> simp [put, h]
  have hn : s.mem.locks next.toInt ≠ (1#32) := hlock
  simp [step, decode, Nat.not_le.mpr hp, hphase, hn, execute, hctl, hput]

theorem failed_poll_forever (b : Backend V) (c : Config) (p : Nat) (next : I32) (s : State V)
    (hp : p < c.grid) (hphase : (s.control p).phase = .poll next)
    (hlock : s.mem.locks next.toInt ≠ 1) (attempts : Nat) :
    run b c (List.replicate attempts p) s =
      (List.replicate attempts (.poll next.toInt), s) := by
  induction attempts with
  | zero => rfl
  | succ n ih =>
    simp [List.replicate_succ, run, failed_poll_step b c p next s hp hphase hlock, ih]

def waitingState := (run fp32 twoPrograms [0,0] (initial fp32 zeroMemory)).2

theorem initial_zero_lock_reaches_wait :
    (waitingState.control 0).phase = .poll 1 ∧ waitingState.mem.locks 1 = 0 := by
  decide +kernel

/-- For every number of polls, the real zero-lock initial condition permits
this nonterminating scheduler prefix. No ready-lock precondition is imposed. -/
theorem consumer_only_schedule_blocks (attempts : Nat) :
    (run fp32 twoPrograms (List.replicate attempts 0) waitingState).2 = waitingState ∧
    (run fp32 twoPrograms (List.replicate attempts 0) waitingState).2.control 0 =
      { round := 1, phase := .poll 1 } := by
  rw [failed_poll_forever fp32 twoPrograms 0 1 waitingState (by decide)
    initial_zero_lock_reaches_wait.1 (by
      change waitingState.mem.locks 1 ≠ 1
      rw [initial_zero_lock_reaches_wait.2]; decide)]
  constructor
  · rfl
  · change waitingState.control 0 = { round := 1, phase := .poll 1 }
    decide +kernel

/-- The concrete prefix publishes once, acquires, then writes P for the second
round before the consumer's first load. The next release has not happened. -/
def reuseSchedule : List Nat :=
  List.replicate 4 1 ++ List.replicate 3 0 ++ List.replicate 4 1 ++ [0]

def reuseTrace := (run fp32 twoPrograms reuseSchedule (initial fp32 zeroMemory)).1

theorem reuse_conflicting_access_prefix :
    reuseTrace = [.silent, .publish 1 0 1, .silent, .release 1,
      .silent, .silent, .poll 1, .output 1 0 1, .silent, .silent,
      .publish 1 0 1, .add 0 0 1] := by decide +kernel

/-- Event-level happens-before edges for this finite prefix. The only
cross-program synchronization is first release (3) -> successful acquire (6).
Program-order edges include every earlier/later event of the same program.
This is a precisely stated trace model, not a formalization of all PTX rules. -/
def reuseEdge (i j : Nat) : Prop :=
  i < j ∧ j < reuseSchedule.length ∧
    (reuseSchedule[i]?.getD 2 = reuseSchedule[j]?.getD 2 ∨ (i = 3 ∧ j = 6))

/-- A simple cut witnesses lack of happens-before: after the acquire there is
no producer-to-consumer edge. Every edge starting at the second P-store stays
on the producer side of the cut (index ≥ 7), where no consumer load occurs. -/
def producerTail (i : Nat) : Prop := 7 ≤ i ∧ i < 11

theorem producerTail_closed {i j : Nat} (hi : producerTail i) (he : reuseEdge i j) :
    producerTail j := by
  rcases hi with ⟨hi7, hi11⟩
  rcases he with ⟨hij, hjlen, hsame | hsync⟩
  · have hj12 : j < 12 := by simpa [reuseSchedule] using hjlen
    interval_cases i <;> interval_cases j <;>
      norm_num [reuseSchedule, producerTail] at *
  · omega

theorem producerTail_trans {i j : Nat} (hi : producerTail i)
    (h : Relation.TransGen reuseEdge i j) : producerTail j := by
  induction h with
  | single he => exact producerTail_closed hi he
  | tail _ he ih => exact producerTail_closed ih he

theorem reuse_happens_before_lt {i j : Nat}
    (h : Relation.TransGen reuseEdge i j) : i < j := by
  induction h with
  | single he => exact he.1
  | tail _ he ih => exact Nat.lt_trans ih he.1

theorem no_happens_before_second_store_to_read :
    ¬ Relation.TransGen reuseEdge 10 11 := by
  intro h
  have hc := producerTail_trans (show producerTail 10 from ⟨by decide, by decide⟩) h
  unfold producerTail at hc
  omega

theorem no_happens_before_read_to_second_store :
    ¬ Relation.TransGen reuseEdge 11 10 := by
  intro h
  have := reuse_happens_before_lt h
  omega

#axiomsClean failed_poll_forever
#axiomsClean consumer_only_schedule_blocks
#axiomsClean reuse_conflicting_access_prefix
#axiomsClean no_happens_before_second_store_to_read
#axiomsClean no_happens_before_read_to_second_store
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock

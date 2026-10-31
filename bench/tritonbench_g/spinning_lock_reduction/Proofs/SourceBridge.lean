import bench.tritonbench_g.spinning_lock_reduction.Proofs.SourceEntry
import Mathlib.Tactic.SplitIfs

/-! Compilation and execution connection for the readable current source.
The plan is computed from the quotation; it is not a second source AST.
CAS executes with its actual old return value; barriers remain explicit in
source traces while projecting to silent controller events.
-/
namespace VeriTile.Bench.Repairs.SpinningLock.SourceReader
open VeriTile.Bench.TritonBenchG.SpinningLock

set_option maxRecDepth 16000
set_option maxHeartbeats 1600000

def sourcePlan : Plan := (compile kernel).get (by decide)

theorem source_compiles : compile kernel = some sourcePlan := by rfl

@[simp] theorem source_start : sourcePlan.start = 1 := rfl
@[simp] theorem source_stop : sourcePlan.stop = 10 := rfl

@[simp] theorem source_reducer (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    evalB h stage p q l sourcePlan.reducer = Wide.reducer64 h p := rfl

@[simp] theorem source_next (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    evalI h stage p q l sourcePlan.nextInitial = BitVec.ofNat 64 p + 1 := rfl

@[simp] theorem source_advance (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    evalI h stage p q l sourcePlan.advance = BitVec.ofNat 64 q + 1 := rfl

@[simp] theorem source_continues (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    evalB h stage p q l sourcePlan.continues = Wide.continues64 h p q := rfl

@[simp] theorem source_poll (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    address h stage p q l sourcePlan.poll = (BitVec.ofNat 64 q).toInt := by
  change 0 + _ = _
  exact Int.zero_add _

@[simp] theorem source_compare (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    word (evalI h stage p q l sourcePlan.compare) = 1 := rfl
@[simp] theorem source_replace (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    word (evalI h stage p q l sourcePlan.replace) = 1 := rfl
@[simp] theorem source_expected (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    word (evalI h stage p q l sourcePlan.expected) = 1 := rfl

@[simp] theorem source_load (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    address h stage p q l sourcePlan.load = Wide.partialOffset64 h q l := by
  change 0 + _ + _ + _ = _ + _ + _
  simp only [Int.zero_add]
  rfl

@[simp] theorem source_publish (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    address h stage p q l sourcePlan.publish = Wide.partialOffset64 h p l := by
  change 0 + _ + _ + _ = _ + _ + _
  simp only [Int.zero_add]
  rfl

@[simp] theorem source_release (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    address h stage p q l sourcePlan.release = (BitVec.ofNat 64 p).toInt := by
  change 0 + _ = _
  exact Int.zero_add _

@[simp] theorem source_releaseWord (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    word (evalI h stage p q l sourcePlan.releaseWord) = 1 := rfl

@[simp] theorem source_output (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    address h stage p q l sourcePlan.output = Wide.outputOffset64 h p l := by
  change 0 + _ + _ = _ + _
  simp only [Int.zero_add]
  rfl

@[simp] theorem source_mask (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    evalB h stage p q l sourcePlan.mask = Wide.outputMask64 h p l := rfl

@[simp] theorem source_assertion (h : HostArgs) (stage : Phased.Stage) :
    evalB h stage 0 0 0 sourcePlan.assertion = Bool.true := by
  cases stage <;> rfl

@[simp] theorem source_skip (h : HostArgs) (stage : Phased.Stage) (p q l : Nat) :
    evalB h stage p q l sourcePlan.skip = !Phased.selected64 h stage p := by
  cases stage
  · change (Bool.true && Wide.reducer64 h p || Bool.false && !Wide.reducer64 h p) = !(!Wide.reducer64 h p)
    simp
  · change (Bool.false && Wide.reducer64 h p || Bool.true && !Wide.reducer64 h p) = !Wide.reducer64 h p
    simp

theorem cas11_preserves (old : I32) : (cas old 1 1).2 = old := by
  simp only [cas]
  split <;> simp_all

theorem execute_cas11 (b : Backend V) (a : Int) (s : State V) :
    executeSource b (.cas a 1 1) s = execute b (.poll a) s := by
  have hp : put s.mem.locks a (s.mem.locks a) = s.mem.locks := by
    funext x
    simp only [put]
    split <;> simp_all
  simp only [executeSource, cas11_preserves, hp, execute]

theorem decode_projection (h : HostArgs) (stage : Phased.Stage) (p : Nat) (s : Wide.State V) :
    (project (decodePlan sourcePlan h stage p s).1, (decodePlan sourcePlan h stage p s).2) =
      Wide.nativeDecode h p s := by
  by_cases hp : p ≥ h.numSms
  · simp [decodePlan, Wide.nativeDecode, hp, project]
  · cases hc : (s.pc p).phase <;>
      simp only [decodePlan, Wide.nativeDecode, if_neg hp, hc, source_stop,
        source_reducer, source_next, source_continues, source_poll, source_compare,
        source_replace, source_expected, source_load, source_advance, source_publish,
        source_release, source_releaseWord, source_output, source_mask, cas] <;>
      (try split_ifs) <;> rfl

theorem decoded_execute (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (p : Nat) (s : Wide.State V) :
    executeSource b (decodePlan sourcePlan h stage p s).1 s.values =
      execute b (project (decodePlan sourcePlan h stage p s).1) s.values := by
  by_cases hp : p ≥ h.numSms
  · simp [decodePlan, hp, executeSource, project]
  · cases hc : (s.pc p).phase <;>
      simp only [decodePlan, if_neg hp, hc, source_stop, source_reducer,
        source_next, source_continues, source_poll, source_compare, source_replace,
        source_expected, source_load, source_advance, source_publish, source_release,
        source_releaseWord, source_output, source_mask, cas] <;>
      (try split_ifs) <;> simp only [project, execute_cas11] <;> rfl

theorem step_projection (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (p : Nat) (s : Wide.State V) :
    (project (stepPlan b sourcePlan h stage p s).1, (stepPlan b sourcePlan h stage p s).2) =
      Wide.nativeStep b h p s := by
  simp only [stepPlan, Wide.nativeStep]
  rw [← decode_projection h stage p s]
  simp only [decoded_execute]

theorem initial_projection (b : Backend V) (mem : Memory V) :
    initialPlan b sourcePlan mem = Wide.initial b mem := rfl

theorem run_projection (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (xs : List Nat) (s : Wide.State V) :
    ((runPlan b sourcePlan h stage xs s).1.map project, (runPlan b sourcePlan h stage xs s).2) =
      Phased.stageRun b h stage xs s := by
  induction xs generalizing s with
  | nil => rfl
  | cons p ps ih =>
    have hp := step_projection b h stage p s
    simp only [runPlan, source_skip, Phased.stageRun, Phased.nativeSchedule, List.filter_cons]
    by_cases hg : p < h.numSms
    · by_cases selected : Phased.selected64 h stage p = Bool.true
      · simp only [hg, show ¬p ≥ h.numSms by omega, decide_false, Bool.false_or,
          selected, Bool.not_true, Bool.false_eq_true, if_false, decide_true,
          Bool.true_and, if_true, Wide.nativeRun, List.map_cons]
        have ih' := ih (stepPlan b sourcePlan h stage p s).2
        rw [Prod.mk.injEq] at hp
        simp only [hp.1, hp.2] at ih' ⊢
        exact congrArg (fun r => ((Wide.nativeStep b h p s).1 :: r.1, r.2)) ih'
      · have hs : Phased.selected64 h stage p = Bool.false := Bool.eq_false_iff.mpr selected
        simpa [hg, hs] using ih s
    · simpa [hg, show p ≥ h.numSms by omega] using ih s

theorem runKernel_projection (b : Backend V) (h : HostArgs) (stage : Phased.Stage)
    (xs : List Nat) (mem : Memory V) :
    Option.map (fun r => (r.1.map project, r.2)) (runKernel b h stage xs mem) =
      some (Phased.stageRun b h stage xs (Wide.initial b mem)) := by
  simp [runKernel, compileAndRun, source_compiles, source_assertion,
    initial_projection, run_projection]

#axiomsClean source_compiles
#axiomsClean decode_projection
#axiomsClean step_projection
#axiomsClean runKernel_projection
#specNonCircular kernel avoiding [Wide.nativeDecode, Wide.nativeRun]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.SpinningLock.SourceReader

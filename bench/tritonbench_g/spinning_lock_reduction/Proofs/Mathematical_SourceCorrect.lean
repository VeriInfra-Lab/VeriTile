import bench.tritonbench_g.spinning_lock_reduction.Proofs.Mathematical_ControlTrace

/-! Source-compiled whole-schedule contracts. The exact full AST-derived typed
source is admitted by a fail-closed restricted compiler, and the resulting plan
executes every guard/address expression. The bridge covers arbitrary schedules.
The Python parser/symbolic substitution and GPU memory refinement boundaries
remain explicit; these are not claims of native race freedom or termination.
-/
namespace VeriTile.Bench.TritonBenchG.SpinningLock.SourceCorrect
open VeriTile.Triton Frontend

/-- The RHS trace is generated from Config, initial locks and schedule alone.
It does NOT require running the source floating-value interpreter first. -/
specification compiled_source_backend {V : Type} (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) :
    let events := controlTrace originalPlan c schedule mem.locks
    ∃ final, compileAndRun originalTypedSource b c schedule mem = some (events, final) ∧
      final.mem = (replay b events (Reference.initial mem)).mem ∧
      (∀ p lane, final.acc p lane =
        ((replay b events (Reference.initial mem)).reads p lane).foldl b.add b.zero) ∧
      (∀ address,
        ((∀ p l, Event.publish p l address ∉ events) → final.mem.partials address = mem.partials address) ∧
        ((∀ p l, Event.output p l address ∉ events) → final.mem.output address = mem.output address) ∧
        (Event.release address ∉ events → final.mem.locks address = mem.locks address)) := by
  dsimp only
  let result := run b c schedule (initial b mem)
  have ht : result.1 = controlTrace originalPlan c schedule mem.locks :=
    original_control_trace b c schedule mem
  rw [← ht]
  refine ⟨result.2, original_source_execution b c schedule mem, ?_, ?_, ?_⟩
  · exact (full_schedule_value_memory b c schedule mem).1
  · exact (full_schedule_value_memory b c schedule mem).2
  · intro address; exact full_schedule_frame b c schedule mem address

specification compiled_source_fp32 (c : Config) (schedule : List Nat)
    (mem : Memory (FP.Value .fp32)) :
    let events := controlTrace originalPlan c schedule mem.locks
    ∃ final, compileAndRun originalTypedSource fp32 c schedule mem = some (events, final) ∧
      final.mem = (replay fp32 events (Reference.initial mem)).mem ∧
      (∀ p lane, final.acc p lane =
        ((replay fp32 events (Reference.initial mem)).reads p lane).foldl
          (FP.add .ieeeValue) (FP.Value.zero .fp32)) ∧
      (∀ address,
        ((∀ p l, Event.publish p l address ∉ events) → final.mem.partials address = mem.partials address) ∧
        ((∀ p l, Event.output p l address ∉ events) → final.mem.output address = mem.output address) ∧
        (Event.release address ∉ events → final.mem.locks address = mem.locks address)) :=
  compiled_source_backend fp32 c schedule mem

/-- Nonempty full nine-round witness now enters through the actual generated
source tree and compiler, rather than entering the hand-written machine. -/
theorem compiled_source_completes_nine_rounds :
    ∃ events final,
      compileAndRun originalTypedSource fp32 twoPrograms producerFirst zeroMemory = some (events, final) ∧
      allDone twoPrograms final = true ∧ final.mem.output 0 = FP.Value.zero .fp32 ∧
      events.count (.add 0 0 1) = 9 ∧ events.count (.publish 1 0 1) = 9 := by
  have he : compileAndRun originalTypedSource fp32 twoPrograms producerFirst zeroMemory =
      some completeRun := original_source_execution fp32 twoPrograms producerFirst zeroMemory
  refine ⟨completeRun.1, completeRun.2, ?_, ?_, ?_, ?_, ?_⟩
  · simpa only [Prod.mk.eta] using he
  · exact nine_rounds_nonempty_execution.1
  · exact nine_rounds_nonempty_execution.2.1
  · exact nine_rounds_nonempty_execution.2.2.2.2.1
  · exact nine_rounds_nonempty_execution.2.2.2.2.2.1

#axiomsClean compiled_source_backend
#axiomsClean compiled_source_fp32
#axiomsClean compiled_source_completes_nine_rounds
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock.SourceCorrect

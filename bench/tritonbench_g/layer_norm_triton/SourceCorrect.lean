import bench.tritonbench_g.layer_norm_triton.Proofs.FrontendChecks

/-! Full generated typed-source contracts for all three kernels. The task-local
Python parser, type/shape checks and compound-node lowering remain explicitly
external trust boundaries. No GPU functional/memory-order/progress claim is made.
The reference trace and numerical observations are independent of source values. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormLock.SourceCorrect
open VeriTile.Triton Frontend
open FP.Equational (ReductionPlan)

specification compiled_forward {V : Type} (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (mem : Memory V) (undef : Undef V)
    (eps : V) (row fuel : Nat) :
    compileForward originalTypedSource b c plan mem undef eps row fuel =
      forwardMemoryReference b c plan mem undef eps row fuel := by
  rw [original_forward_execution, forward_kernel_value]

specification compiled_backward {V : Type} (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (inputs : Memory V) (undef : Undef V)
    (mem : Shared V) (words : Int → I32) (schedule : List Nat) :
    let rows := backwardReference b c plan inputs undef
    let events := controlTrace c schedule words
    ∃ finalState, compileBackward originalTypedSource b c plan inputs undef mem words schedule =
        some (events, finalState) ∧
      Related b rows finalState.values (replay b c rows undef events (Reference.initial mem)) ∧
      (∀ region address fallback, (∀ e ∈ events, Untouched e region address) →
        finalState.values.mem.read region address fallback = mem.read region address fallback) := by
  dsimp only
  let sourceRows := backwardSource b c plan inputs undef
  let result := run b c sourceRows undef schedule (initial sourceRows mem words)
  have h := ConcurrentCorrect.full_backward b c plan inputs undef mem words schedule
  refine ⟨result.2, ?_, h.2, ?_⟩
  · rw [original_backward_execution]
    change some result = some (controlTrace c schedule words, result.2)
    congr 1
    exact Prod.ext h.1 rfl
  · intro region address fallback untouched
    exact ConcurrentCorrect.full_backward_frame b c plan inputs undef mem words schedule region address fallback untouched

specification compiled_final_dwdb {V : Type} (b : Backend V) (c : Config) (mem : Memory V)
    (m : I32) (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) :
    compileFinal originalTypedSource b c mem m bm bn pid plan fuel =
      finalMemoryReference b c mem m bm bn pid plan fuel := by
  rw [original_final_execution, final_kernel_value]

/-- Every used scalar and storage operation is closed software bits here;
sqrt is not called by the complete backward kernel. The explicit reduction
plan and source-order arithmetic profile still require native binding. -/
specification compiled_backward_software (c : Config) (plan : ReductionPlan c.block)
    (inputs : Memory Word) (undef : Undef Word) (mem : Shared Word)
    (words : Int → I32) (schedule : List Nat) :
    let rows := backwardReference backwardBits c plan inputs undef
    let events := controlTrace c schedule words
    ∃ finalState, compileBackward originalTypedSource backwardBits c plan inputs undef mem words schedule =
        some (events, finalState) ∧
      Related backwardBits rows finalState.values
        (replay backwardBits c rows undef events (Reference.initial mem)) ∧
      (∀ region address fallback, (∀ e ∈ events, Untouched e region address) →
        finalState.values.mem.read region address fallback = mem.read region address fallback) :=
  compiled_backward backwardBits c plan inputs undef mem words schedule

theorem final_sqrt_unused (sqrt1 sqrt2 : Word → Word) (c : Config) (mem : Memory Word)
    (m : I32) (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) :
    finalKernel (software sqrt1) c mem m bm bn pid plan fuel =
      finalKernel (software sqrt2) c mem m bm bn pid plan fuel := by
  rw [final_kernel_value, final_kernel_value]
  unfold finalMemoryReference finalReference
  simp only [sum, reduce_software]
  rfl

specification compiled_final_software (c : Config) (mem : Memory Word)
    (m : I32) (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) :
    compileFinal originalTypedSource backwardBits c mem m bm bn pid plan fuel =
      finalMemoryReference backwardBits c mem m bm bn pid plan fuel :=
  compiled_final_dwdb backwardBits c mem m bm bn pid plan fuel

/-- Nonempty original typed-source entry: both full critical sections finish,
including first Count==0 publication and later rounded accumulation. -/
theorem source_two_rows_complete_and_round :
    ∃ events finalState,
      compileBackward originalTypedSource backwardBits twoRows oneLane exampleInputs exampleUndef
        initialShared (fun _ => 0) orderedSchedule = some (events, finalState) ∧
      finalState.control.pc 0 = .done ∧ finalState.control.pc 1 = .done ∧
      finalState.values.mem.db 0 = big ∧ finalState.values.mem.dx 2 = big := by
  have he : compileBackward originalTypedSource backwardBits twoRows oneLane exampleInputs exampleUndef
        initialShared (fun _ => 0) orderedSchedule = some completeRun :=
    original_backward_execution backwardBits twoRows oneLane exampleInputs exampleUndef initialShared (fun _ => 0) orderedSchedule
  refine ⟨completeRun.1, completeRun.2, ?_, two_rows_complete_and_round.1,
    two_rows_complete_and_round.2.1, two_rows_complete_and_round.2.2.2.2.1,
    two_rows_complete_and_round.2.2.2.2.2.2.2.2.1⟩
  simpa only [Prod.mk.eta] using he

#axiomsClean compiled_forward
#axiomsClean compiled_backward
#axiomsClean compiled_final_dwdb
#axiomsClean compiled_backward_software
#axiomsClean source_two_rows_complete_and_round
#axiomsClean final_sqrt_unused
#axiomsClean compiled_final_software
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock.SourceCorrect

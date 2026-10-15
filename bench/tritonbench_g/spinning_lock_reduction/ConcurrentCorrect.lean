import bench.tritonbench_g.spinning_lock_reduction.Proofs.Progress

/-! Public contracts for the FULL task-local SC control model. These declarations
are intentionally named ConcurrentCorrect, not a standard DSL source-admission
entry. Neither native termination nor race freedom nor GPU bit conformance has
been proved. A shared while/atomic-expression elaboration bridge is still absent.
-/
namespace VeriTile.Bench.TritonBenchG.SpinningLock.ConcurrentCorrect
open VeriTile.Triton

/-- Every complete or incomplete schedule is specified. The result includes
all shared regions and an independent ordered fold for every accumulator. -/
specification whole_schedule_backend {V : Type} (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) :
    let result := run b c schedule (initial b mem)
    let reference := replay b result.1 (Reference.initial mem)
    result.2.mem = reference.mem ∧ ∀ p lane,
      result.2.acc p lane = (reference.reads p lane).foldl b.add b.zero :=
  full_schedule_value_memory b c schedule mem

/-- Closed fp32 software interpretation of exactly the same full scheduler;
no algebraic laws, callback remainder, finite-input or ready-lock hypothesis. -/
specification whole_schedule_fp32 (c : Config) (schedule : List Nat)
    (mem : Memory (FP.Value .fp32)) :
    let result := run fp32 c schedule (initial fp32 mem)
    let reference := replay fp32 result.1 (Reference.initial mem)
    result.2.mem = reference.mem ∧ ∀ p lane,
      result.2.acc p lane =
        (reference.reads p lane).foldl (FP.add .ieeeValue) (FP.Value.zero .fp32) :=
  full_schedule_value_memory fp32 c schedule mem

/-- Every untouched address in each source memory region preserves its bits. -/
specification whole_schedule_frame {V : Type} (b : Backend V) (c : Config)
    (schedule : List Nat) (mem : Memory V) (address : Int) :
    let result := run b c schedule (initial b mem)
    ((∀ p l, Event.publish p l address ∉ result.1) → result.2.mem.partials address = mem.partials address) ∧
    ((∀ p l, Event.output p l address ∉ result.1) → result.2.mem.output address = mem.output address) ∧
    (Event.release address ∉ result.1 → result.2.mem.locks address = mem.locks address) :=
  full_schedule_frame b c schedule mem address

#axiomsClean whole_schedule_backend
#axiomsClean whole_schedule_fp32
#axiomsClean whole_schedule_frame
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SpinningLock.ConcurrentCorrect

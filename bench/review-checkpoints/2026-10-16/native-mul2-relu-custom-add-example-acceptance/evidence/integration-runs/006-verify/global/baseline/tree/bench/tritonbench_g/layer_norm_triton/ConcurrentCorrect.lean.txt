import bench.tritonbench_g.layer_norm_triton.Proofs.Software
import bench.tritonbench_g.layer_norm_triton.Proofs.FrameProgress

/-! Complete three-kernel value contracts for the explicitly defined task-local
SC source model. These declarations do not assert shared-DSL source admission,
GPU race freedom, GPU progress, or native sqrt/rounding correspondence. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormLock.ConcurrentCorrect
open VeriTile.Triton
open FP.Equational (ReductionPlan)

specification full_forward {V : Type} (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (mem : Memory V) (undef : Undef V)
    (eps : V) (row fuel : Nat) :
    forwardKernel b c plan mem undef eps row fuel =
      forwardMemoryReference b c plan mem undef eps row fuel :=
  forward_kernel_value b c plan mem undef eps row fuel

/-- All source row arithmetic is evaluated, then all DX stores, failed CAS
retries, first-count branch, partial loads/adds/narrow stores and releases are
scheduled. The reference has independent control and independent observation
memory, with no supplied intermediate numerical results or assumed steps. -/
specification full_backward {V : Type} (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (inputs : Memory V) (undef : Undef V)
    (mem : Shared V) (words : Int → I32) (schedule : List Nat) :
    let sourceRows := backwardSource b c plan inputs undef
    let refRows := backwardReference b c plan inputs undef
    let events := controlTrace c schedule words
    let actual := run b c sourceRows undef schedule (initial sourceRows mem words)
    actual.1 = events ∧ Related b refRows actual.2.values
      (replay b c refRows undef events (Reference.initial mem)) := by
  dsimp only
  have hr : backwardSource b c plan inputs undef = backwardReference b c plan inputs undef :=
    funext (backward_value b c plan inputs undef)
  rw [hr]
  exact whole_schedule b c (backwardReference b c plan inputs undef) undef schedule mem words

specification full_backward_frame {V : Type} (b : Backend V) (c : Config)
    (plan : ReductionPlan c.block) (inputs : Memory V) (undef : Undef V)
    (mem : Shared V) (words : Int → I32) (schedule : List Nat)
    (region : Region) (address : Int) (fallback : V)
    (h : ∀ e ∈ controlTrace c schedule words, Untouched e region address) :
    let rows := backwardSource b c plan inputs undef
    (run b c rows undef schedule (initial rows mem words)).2.values.mem.read region address fallback =
      mem.read region address fallback := by
  dsimp only
  apply run_frame
  rw [(whole_schedule b c (backwardSource b c plan inputs undef) undef schedule mem words).1]
  exact h

specification full_final_dwdb {V : Type} (b : Backend V) (c : Config) (mem : Memory V)
    (m : I32) (bm bn pid : Nat) (plan : ReductionPlan bm) (fuel : Nat) :
    finalKernel b c mem m bm bn pid plan fuel =
      finalMemoryReference b c mem m bm bn pid plan fuel :=
  final_kernel_value b c mem m bm bn pid plan fuel

/-- Closed scalar and fp16/bf16/fp32 storage bit interpreter for the complete
backward schedule. The unused sqrt slot is proved irrelevant in Software. -/
specification full_backward_software (c : Config) (plan : ReductionPlan c.block)
    (inputs : Memory Word) (undef : Undef Word) (mem : Shared Word)
    (words : Int → I32) (schedule : List Nat) :
    let sourceRows := backwardSource backwardBits c plan inputs undef
    let refRows := backwardReference backwardBits c plan inputs undef
    let events := controlTrace c schedule words
    let actual := run backwardBits c sourceRows undef schedule (initial sourceRows mem words)
    actual.1 = events ∧ Related backwardBits refRows actual.2.values
      (replay backwardBits c refRows undef events (Reference.initial mem)) :=
  full_backward backwardBits c plan inputs undef mem words schedule

#axiomsClean full_forward
#axiomsClean full_backward
#axiomsClean full_backward_frame
#axiomsClean full_final_dwdb
#axiomsClean full_backward_software
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LayerNormLock.ConcurrentCorrect

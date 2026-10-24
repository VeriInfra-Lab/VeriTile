import bench.repairs.tritonbench_g.kldiv_ops.Misc.RealBackward
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.SourceBytes
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.NativeReport
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.AllFrames
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.FailureTrace
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.LibraryWitness

/-! Whole repaired FP32 wrapper: independent Real target and actual physical IO.
Forward prediction is [BT,V]; the target may have any shape/rank with enough
logical elements. Backward target is [BT,V]; destination may have any shape/rank
with enough elements. These capacities are checked in source after preparations.
All legal strides, offsets, lazy negative flags, aliases and empty axes are
represented. Backward copies a fresh dense work buffer into the original
logical destination; a scalar-one cotangent returns that same object, otherwise
Torch broadcasting determines the returned shape. Oversized destination tails
come from the actual empty allocation, not an invented derivative.

The independent Real loss is t*(log(max(t,eps))-y), or exp(t)*(t-y), with the
four actual reductions. Backward is the prediction derivative times the entry
cotangent; normalization belongs to that cotangent. Real division by zero is
totalized mathematically and is not asserted to equal exceptional FP behavior.
No Codec Real, whole-output premise, finite-size cap or kernel-success oracle
occurs. Native runtime grid admission remains a local dispatch deferral.
-/
namespace VeriTile.Bench.Repairs.KldivOps.Correctness
open Storage Host VeriTile.Bench.Repairs.DiagSsmTriton.ByteMemory
open VeriTile.Triton.FP.Structural (Algebra)
noncomputable section

abbrev realLoss := @RealExecution.loss
abbrev realForward := @RealExecution.forward
abbrev realBackward := @RealExecution.backward
abbrev predictionDerivative := @RealExecution.loss_increment
abbrev completeRealForward := @RealExecution.returned_forward_ideal
abbrev completeRealBackwardPrefix := @RealExecution.returned_backward_prefix_ideal
abbrev completeRealBackward := @RealExecution.returned_backward_ideal

/-- Success yields the complete source trace, entry-value binding, actual
returned object/reduction and every caller byte frame. Capacity follows from
the source guard; it is not a theorem precondition. -/
theorem forward {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (lib : Library codec num api) (compiler : Compiler)
    (prediction target : Tensor) (flag : Bool) (name : String) (eps : Word)
    (entry : State) (returned : Tensor) (final : State)
    (predLive : prediction.backing∈entry.live) (targetLive : target.backing∈entry.live)
    (call : Host.forward api (sourceActions codec M compiler) prediction target flag name eps entry=.ok returned final) :
    ∃ f : ForwardResult codec num M api compiler prediction target flag name eps entry returned final,
      prediction.numel ≤ target.numel ∧ ∀ i,i<returned.numel →
        value codec num.neg final.heap returned i=reducedWord num f.trace.red f.trace.rows f.trace.cols
          (forwardKernelValue M f.trace.red f.trace.cols flag eps
            (value codec num.neg entry.heap prediction) (value codec num.neg entry.heap target)) i := by
  obtain ⟨f⟩ := forward_correct codec num M api lib compiler prediction target flag name eps entry returned final predLive targetLive call
  exact ⟨f,f.capacity,f.returned_values f.capacity⟩

theorem backward {Word : Type} [Inhabited Word] (codec : Codec Word) (num : Numerics Word)
    (M : Algebra Word) (api : Services Word) (lib : Library codec num api) (compiler : Compiler)
    (target grad dst : Tensor) (flag : Bool) (entry : State) (returned : Tensor) (final : State)
    (tlive : target.backing∈entry.live) (glive : grad.backing∈entry.live) (dlive : dst.backing∈entry.live)
    (call : Host.backward api (sourceActions codec M compiler) target grad dst flag entry=.ok returned final) :
    ∃ f : BackwardResult codec num M api compiler target grad dst flag entry returned final,
      target.numel ≤ dst.numel ∧ ∀ i,i<dst.numel → value codec num.neg final.heap dst i=
        backwardKernelValue M flag target.numel (value codec num.neg entry.heap target)
          (value codec num.neg f.trace.ready.heap f.trace.work) i := by
  obtain ⟨f⟩ := backward_correct codec num M api lib compiler target grad dst flag entry returned final tlive glive dlive call
  exact ⟨f,f.capacity,f.destination_values f.capacity⟩

abbrev backwardReturnedValues := @BackwardResult.returned_values
abbrev forwardAllOutcomes := @forward_all_frames
abbrev backwardAllOutcomes := @backward_all_frames
abbrev forwardCapacityFailure := @forward_capacity_failure
abbrev backwardCapacityFailure := @backward_capacity_failure
abbrev backwardFailureStages := @backward_finish_failure
abbrev forwardPhysicalStores := @ForwardResult.source_bytes
abbrev backwardPhysicalStores := @BackwardResult.source_bytes
abbrev fullFP32ByteRoundtrip := @fp32_lossless
abbrev nativeIntegers := @dense_intermediates
abbrev forwardPointers := @ForwardResult.native_pointers
abbrev backwardPointers := @BackwardResult.native_pointers
abbrev consistentWholeCall := @Witness.whole_forward_inhabited

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Correctness

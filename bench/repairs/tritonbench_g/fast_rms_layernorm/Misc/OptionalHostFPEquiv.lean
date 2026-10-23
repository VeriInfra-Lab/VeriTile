import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.NativeHost
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPCurrent
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.CheckedHost
import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPHost
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostStorage
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostIdentity
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.FPPhysicalFrame

/-! Checked contiguous FP32 CUDA source entry. Forward X/W are direct views;
only Y/r are freshly allocated. Backward copies G into private dX because the
original dX kernel writes in place, then computes independent dW before dX.
Combined adjoints require unchanged saved inputs at successful version retrieval.
The source-only theorem is also exposed for internal mathematical reuse. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.OptionalHostFPEquiv
open VeriTile.Triton FP.Structural Host
noncomputable section
variable {α : Type} [Inhabited α]

def defaultPrecision : ComputeDType := .fp32
abbrev forwardReference := @FPReference.output
abbrev savedReference := @FPReference.savedInv
abbrev inputGradientReference := @FPBackwardReference.output
abbrev weightGradientReference := @FPWeightReference.output

/-- Exact local interpretation of torch's FP32 zero_ storage primitive. -/
def StorageZero (zero : Cell α) (encode : Cell α → Fin 4 → Fin 256) : Prop :=
  ∀ byte,encode zero byte=0

theorem source_wrapper_equiv {β : Type} (M : Algebra α) (r : Request (Cell α)) (h : Call) (scratch : FPGridForward.Heap α)
    (live : Caller) (encode : Cell α → Fin 4 → β) (entry : Storage.Heap β)
    (prepare : ∀ k,Preparation k .forward live) (p : Prepared h .forward live)
    (valid : validate r=.ok h) (ready : prepare h=.ok p) (zero : Cell α) :
    ∃ actualForward,
      Host.forward (FPGridForward.run M) r scratch live encode entry prepare (.ok ())=
        finish p encode entry actualForward
          (.ok (makeContext r h p actualForward,outputTensor r h p actualForward)) ∧
      FPHost.ForwardSpec M r h scratch actualForward ∧
      ∀ (g : Upstream (Cell α)) (backwardLive : Caller) (backwardEntry : Storage.Heap β)
        (bp : BackwardPrepared (makeContext r h p actualForward) backwardLive),
        backwardValidate (makeContext r h p actualForward) g=.ok () →
        ∃ actualBackward,
          Host.backward (FPGridWeight.run M) (FPGridBackward.run M) zero
            (makeContext r h p actualForward) g backwardLive encode backwardEntry (.ok bp) (.ok ()) (.ok ()) (.ok ())=
            finish bp.toPrepared encode backwardEntry actualBackward
              (.ok (gradientTensors (makeContext r h p actualForward) bp actualBackward)) ∧
          FPHost.BackwardSpec M r h scratch g zero actualBackward := by
  obtain ⟨actual,hostRun,sourceRun,correct⟩:=FPHost.forward_complete M r h scratch live encode entry prepare p valid ready
  refine ⟨actual,hostRun,correct,?_⟩
  intro g backwardLive backwardEntry bp validBackward
  exact FPHost.backward_complete M r h scratch actual live p sourceRun g zero backwardLive encode backwardEntry bp validBackward


theorem wrapper_equiv {β : Type} (M : Algebra α) (r : Request (Cell α)) (m : Contract.Metadata) (admitted : Contract.Admitted r m) (scratch : FPGridForward.Heap α)
    (live : Caller) (encode : Cell α → Fin 4 → β) (entry : Storage.Heap β)
    (prepare : ∀ k,Preparation k .forward live) (p : Prepared (Contract.checkedCall r m admitted) .forward live)
    (ready : prepare (Contract.checkedCall r m admitted)=.ok p)
    (views : CheckedHost.DirectViews r m (Contract.checkedCall r m admitted) live p)
    (inputBytes : CheckedHost.InputsAtEntry r (Contract.checkedCall r m admitted) live p encode entry) (zero : Cell α) :
    let h:=Contract.checkedCall r m admitted
    ∃ actualForward,
      CheckedHost.forward (FPGridForward.run M) r m scratch live encode entry prepare (.ok ())=
        finish p encode entry actualForward
          (.ok (makeContext r h p actualForward,outputTensor r h p actualForward)) ∧
      FPHost.ForwardSpec M r h scratch actualForward ∧
      ∀ (g : Upstream (Cell α)) (gm : Contract.TensorMetadata) (backwardLive : Caller) (backwardEntry : Storage.Heap β)
        (bp : BackwardPrepared (makeContext r h p actualForward) backwardLive),
        Contract.BackwardAdmitted (makeContext r h p actualForward) g gm →
        HostStorage.SavedAtEntry (makeContext r h p actualForward) encode backwardEntry →
        ∃ actualBackward,
          CheckedHost.backward (FPGridWeight.run M) (FPGridBackward.run M) zero
            (makeContext r h p actualForward) g gm backwardLive encode backwardEntry (.ok bp) (.ok ()) (.ok ()) (.ok ())=
            finish bp.toPrepared encode backwardEntry actualBackward
              (.ok (gradientTensors (makeContext r h p actualForward) bp actualBackward)) ∧
          FPHost.BackwardSpec M r h scratch g zero actualBackward := by
  let h:=Contract.checkedCall r m admitted
  have valid:=Contract.checked_valid r m admitted

  obtain ⟨actual,hostRun,sourceRun,correct⟩:=FPHost.forward_complete M r h scratch live encode entry prepare p valid ready
  refine ⟨actual,?_,correct,?_⟩
  · rw [CheckedHost.forward_admitted _ r m admitted]
    exact hostRun
  · intro g gm backwardLive backwardEntry bp backwardOK currentSaved
    rw [CheckedHost.backward_admitted _ _ _ _ g gm backwardOK]
    have validBackward:=Contract.backward_validated _ g gm backwardOK
    exact FPHost.backward_complete M r h scratch actual live p sourceRun g zero backwardLive encode backwardEntry bp validBackward

/-- Native scalar instruction graphs; reduction trees and launch-specific
fusion are compiler structure, not numerical ports. -/
abbrev native_forward := NativeHost.forward
abbrev native_backward := NativeHost.backward
abbrev native_reduction_coverage := @NativeGraph.reduction_complete
abbrev native_weight_fusion_disabled := NativeGraph.weight_fusion_disabled
abbrev native_compiler_expression := @NativeGraph.compiled_expression

def nativeForwardReference (R : NativeGraph.Reductions) (S : NativeGraph.Selection)
    (I : NativeGraph.Instructions) (h : Call) (x w : Fin h.block → NativeGraph.Term) (j : Fin h.block) :=
  NativeGraph.observe I S (FPReference.output (NativeGraph.algebra R .forward) h x w j)
def nativeSavedReference (R : NativeGraph.Reductions) (S : NativeGraph.Selection)
    (I : NativeGraph.Instructions) (h : Call) (x : Fin h.block → NativeGraph.Term) :=
  NativeGraph.observe I S (FPReference.savedInv (NativeGraph.algebra R .forward) h x)
def nativeInputGradientReference (R : NativeGraph.Reductions) (S : NativeGraph.Selection)
    (I : NativeGraph.Instructions) (h : Call) (x w g : Fin h.block → NativeGraph.Term)
    (saved : NativeGraph.Term) (j : Fin h.block) :=
  NativeGraph.observe I S (FPBackwardReference.output (NativeGraph.algebra R .input) h x w g saved j)
def nativeWeightGradientReference (R : NativeGraph.Reductions) (S : NativeGraph.Selection)
    (I : NativeGraph.Instructions) (gemma : Bool) (x g saved : Nat → NativeGraph.Term) (rows : Nat) :=
  NativeGraph.observe I S (FPWeightReference.output (NativeGraph.algebra R .weight) gemma x g saved rows)

#axiomsClean native_forward
#axiomsClean native_backward
#specNonCircular nativeForwardReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward,FP.Structural.exec]
#specNonCircular nativeSavedReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward,FP.Structural.exec]
#specNonCircular nativeInputGradientReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward,FP.Structural.exec]
#specNonCircular nativeWeightGradientReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward,FP.Structural.exec]

abbrev backward_current_saved := @FPCurrent.backward_current

abbrev forward_caller_bytes := @CheckedHost.forward_frame
abbrev backward_caller_bytes := @CheckedHost.backward_frame
abbrev preparation_can_fail := Host.preparation_failure
abbrev complete_shapes := @HostStorage.returned_shapes
abbrev forward_byte_readback := @HostStorage.forward_readback
abbrev backward_byte_readback := @HostStorage.backward_readback
abbrev unchanged_saved_bytes := @HostStorage.saved_survives
abbrev saved_raw_backing_bytes := @HostStorage.backward_saved_bytes
abbrev saved_backing_all_outcomes := @HostStorage.backward_saved_full_frame
abbrev saved_current_entry_readback := @HostStorage.backward_current_saved_readback
abbrev forward_source_byte_frame := @FPPhysicalFrame.forward
abbrev backward_source_byte_frame := @FPPhysicalFrame.backward
abbrev fresh_output_identity := @HostIdentity.forward_fresh
abbrev fresh_gradient_identities := @HostIdentity.gradients_fresh
abbrev saved_storage_identity := @HostIdentity.saved_identity
abbrev direct_input_identity := @CheckedHost.saved_input_identity
abbrev python_checks_iff := @Contract.python_checks_iff
abbrev physical_addresses := NativeGrid.physical_bytes
abbrev row_integer_operations := @CheckedHost.checked_row_words
abbrev weight_integer_operations := NativeGrid.weight_row_words
abbrev loop_integer_operations := NativeGrid.loop_words

#axiomsClean wrapper_equiv
#specNonCircular forwardReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#specNonCircular savedReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#specNonCircular inputGradientReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#specNonCircular weightGradientReference avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#auditModuleAxioms
#print axioms wrapper_equiv
-- #print_fp_assumptions accepts the equivalence-calculus head, not this full
-- wrapper-outcome proposition. The audit above checks the actual theorem;
-- it has no numerical atomic-law premise.
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.OptionalHostFPEquiv

import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealCurrent
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.CheckedHost
import bench.repairs.tritonbench_g.fast_rms_layernorm.Kernels
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealHost
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostStorage
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.StorageAccess
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.HostIdentity
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RealPhysicalFrame

/-! Checked contiguous FP32 CUDA source entry. Forward X/W are direct views;
only Y/r are freshly allocated. Backward copies G into private dX because the
original dX kernel writes in place, then computes independent dW before dX.
Combined adjoints require unchanged saved inputs at successful version retrieval.
The source-only theorem is also exposed for internal mathematical reuse. -/
namespace VeriTile.Bench.Repairs.FastRmsLayernorm.OptionalHostCorrect
open VeriTile Triton Host
noncomputable section

abbrev output := @RealReference.output
abbrev inputGradient := @RealReference.inputGradient
abbrev weightGradient := @RealReference.weightGradient

theorem source_wrapper_correct {β : Type} (r : Request MemCell) (h : Call) (scratch : RealGridForward.Heap)
    (live : Caller) (encode : MemCell → Fin 4 → β) (entry : Storage.Heap β)
    (prepare : ∀ k,Preparation k .forward live) (p : Prepared h .forward live)
    (valid : validate r=.ok h) (ready : prepare h=.ok p) : Spec.Real (
    ∃ actualForward,
      Host.forward RealGridForward.run r scratch live encode entry prepare (.ok ())=
        finish p encode entry actualForward
          (.ok (makeContext r h p actualForward,outputTensor r h p actualForward)) ∧
      RealHost.ForwardSpec r h actualForward ∧
      ∀ (g : Upstream MemCell) (backwardLive : Caller) (backwardEntry : Storage.Heap β)
        (bp : BackwardPrepared (makeContext r h p actualForward) backwardLive),
        backwardValidate (makeContext r h p actualForward) g=.ok () →
        ∃ actualBackward,
          Host.backward RealGridWeight.run RealGridBackward.run (.real 0)
            (makeContext r h p actualForward) g backwardLive encode backwardEntry (.ok bp) (.ok ()) (.ok ()) (.ok ())=
            finish bp.toPrepared encode backwardEntry actualBackward
              (.ok (gradientTensors (makeContext r h p actualForward) bp actualBackward)) ∧
          RealHost.BackwardSpec r h g actualBackward) := by
  obtain ⟨actual,hostRun,sourceRun,correct⟩:=RealHost.forward_complete r h scratch live encode entry prepare p valid ready
  refine ⟨actual,hostRun,correct,?_⟩
  intro g backwardLive backwardEntry bp validBackward
  exact RealHost.backward_complete r h scratch actual live p sourceRun g backwardLive encode backwardEntry bp validBackward


theorem wrapper_correct {β : Type} (r : Request MemCell) (m : Contract.Metadata) (admitted : Contract.Admitted r m) (scratch : RealGridForward.Heap)
    (live : Caller) (encode : MemCell → Fin 4 → β) (entry : Storage.Heap β)
    (prepare : ∀ k,Preparation k .forward live) (p : Prepared (Contract.checkedCall r m admitted) .forward live)
    (ready : prepare (Contract.checkedCall r m admitted)=.ok p)
    (views : CheckedHost.DirectViews r m (Contract.checkedCall r m admitted) live p)
    (inputBytes : CheckedHost.InputsAtEntry r (Contract.checkedCall r m admitted) live p encode entry) : let h:=Contract.checkedCall r m admitted; Spec.Real (
    ∃ actualForward,
      CheckedHost.forward RealGridForward.run r m scratch live encode entry prepare (.ok ())=
        finish p encode entry actualForward
          (.ok (makeContext r h p actualForward,outputTensor r h p actualForward)) ∧
      RealHost.ForwardSpec r h actualForward ∧
      ∀ (g : Upstream MemCell) (gm : Contract.TensorMetadata) (backwardLive : Caller) (backwardEntry : Storage.Heap β)
        (bp : BackwardPrepared (makeContext r h p actualForward) backwardLive),
        Contract.BackwardAdmitted (makeContext r h p actualForward) g gm →
        HostStorage.SavedAtEntry (makeContext r h p actualForward) encode backwardEntry →
        ∃ actualBackward,
          CheckedHost.backward RealGridWeight.run RealGridBackward.run (.real 0)
            (makeContext r h p actualForward) g gm backwardLive encode backwardEntry (.ok bp) (.ok ()) (.ok ()) (.ok ())=
            finish bp.toPrepared encode backwardEntry actualBackward
              (.ok (gradientTensors (makeContext r h p actualForward) bp actualBackward)) ∧
          RealHost.BackwardSpec r h g actualBackward) := by
  let h:=Contract.checkedCall r m admitted
  have valid:=Contract.checked_valid r m admitted

  obtain ⟨actual,hostRun,sourceRun,correct⟩:=RealHost.forward_complete r h scratch live encode entry prepare p valid ready
  refine ⟨actual,?_,correct,?_⟩
  · rw [CheckedHost.forward_admitted _ r m admitted]
    exact hostRun
  · intro g gm backwardLive backwardEntry bp backwardOK currentSaved
    rw [CheckedHost.backward_admitted _ _ _ _ g gm backwardOK]
    have validBackward:=Contract.backward_validated _ g gm backwardOK
    exact RealHost.backward_complete r h scratch actual live p sourceRun g backwardLive encode backwardEntry bp validBackward

abbrev backward_current_saved := @RealCurrent.backward_current

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
abbrev saved_allocation_addresses := @Host.saved_addresses
abbrev forward_source_byte_frame := @RealPhysicalFrame.forward
abbrev backward_source_byte_frame := @RealPhysicalFrame.backward
abbrev fresh_output_identity := @HostIdentity.forward_fresh
abbrev fresh_gradient_identities := @HostIdentity.gradients_fresh
abbrev saved_storage_identity := @HostIdentity.saved_identity
abbrev direct_input_identity := @CheckedHost.saved_input_identity
abbrev python_checks_iff := @Contract.python_checks_iff
abbrev finite_forward := StorageAccess.forward_flat
abbrev finite_backward := StorageAccess.backward_flat
abbrev finite_weight_backward := StorageAccess.weight_flat
abbrev physical_addresses := NativeGrid.physical_bytes
abbrev row_integer_operations := @CheckedHost.checked_row_words
abbrev weight_integer_operations := NativeGrid.weight_row_words
abbrev loop_integer_operations := NativeGrid.loop_words
abbrev input_derivative := @RealReference.row_loss_input_directional
abbrev weight_derivative := @RealReference.full_weight_directional

#axiomsClean wrapper_correct
#specNonCircular output avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#specNonCircular inputGradient avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#specNonCircular weightGradient avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward,Host.forward,Host.backward]
#auditModuleAxioms
#print axioms wrapper_correct
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.OptionalHostCorrect

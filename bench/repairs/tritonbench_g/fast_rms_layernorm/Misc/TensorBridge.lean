import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RawCorrect
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RawNative
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.PhysicalIO

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.TensorBridge

/-- The existing Python checks admit exactly the forward and backward contracts. -/
alias python_checks_iff := Contract.python_checks_iff
alias backward_checks_iff := Contract.backward_checks_iff

/-- Shared kernel offsets identify the corresponding actual Tensor bytes. -/
alias physical_addresses := NativeGrid.physical_bytes

#axiomsClean python_checks_iff
#axiomsClean backward_checks_iff
#axiomsClean physical_addresses
noncomputable section

section Mathematical
open VeriTile Triton KernelTensor KernelStorage RawCorrect

theorem real_kernel_forward {β : Type} (h : Call) (storage : Placement h .forward)
    (tensors : RegionName → Torch.TensorMetadata) (binding : Binding h .forward storage tensors)
    (heap : Nat → β) (decode : (Fin 4 → β) → MemCell) (other : MemCell)
    (logical : RegionName → Nat → ℝ) (reads : ReadLaw .forward tensors heap decode RealGridForward.read logical) :
    Spec.Real (∃ final,RealGridForward.run h (lift storage heap decode other)=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,RealGridForward.read (final ("y",i.val*h.cols+j.val))=
        output h.gemma h.eps (rowValues h logical "x" i) (weightValues h logical) j) ∧
      (∀ i : Fin h.rows,RealGridForward.read (final ("r",i.val))=RealReference.inv h.eps (rowValues h logical "x" i)) ∧
      Frame h .forward (lift storage heap decode other) final) :=
  RealTensor.forward h storage tensors binding heap decode other logical reads

/-- The saved statistic is the one proved by kernel_forward, with unchanged
saved X. For zero rows the selected source skips launches and zero-fills dW. -/
theorem real_kernel_backward {β : Type} (h : Call) (storage : Placement h .backward)
    (tensors : RegionName → Torch.TensorMetadata) (binding : Binding h .backward storage tensors)
    (heap : Nat → β) (decode : (Fin 4 → β) → MemCell) (other : MemCell)
    (logical : RegionName → Nat → ℝ) (reads : ReadLaw .backward tensors heap decode RealGridForward.read logical)
    (saved : ∀ i : Fin h.rows,logical "r" i.val=RealReference.inv h.eps (rowValues h logical "x" i)) :
    Spec.Real (∃ final,RealTensor.backwardRun h (lift storage heap decode other)=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,RealGridForward.read (final ("dx",i.val*h.cols+j.val))=
        inputGradient h.gemma h.eps (rowValues h logical "x" i) (weightValues h logical) (rowValues h logical "dx" i) j) ∧
      (∀ j : Fin h.cols,RealGridForward.read (final ("dw",j.val))=
        weightGradient h.eps (rowValues h logical "x") (rowValues h logical "dx") j) ∧
      Frame h .backward (lift storage heap decode other) final) :=
  RealTensor.backward h storage tensors binding heap decode other logical reads saved

end Mathematical

section Native
open VeriTile Triton FP.Structural KernelTensor KernelStorage NativeGraph FPEquiv RawNative

theorem native_kernel_forward (R : Reductions) (S : Selection) (I : Instructions)
    (h : Call) (epsilon : EpsilonABI.Binding h.eps) (storage : Placement h .forward) (tensors : RegionName → Torch.TensorMetadata)
    (binding : Binding h .forward storage tensors) (heap : Nat → Fin 256)
    (other : Cell Term) (logical : RegionName → Nat → Word) (reads : WordReadLaw .forward tensors heap logical) :
    ∃ final,RuntimeForward.Grid.run (algebra R .forward) h (.epsilon epsilon.runtime) (lift storage heap decodeBytes other)=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,∀ byte : Fin 4,
        commit storage (encode I S) heap (fun r o => final (r,o))
          (address storage "y" (by simp [buffers]) (i.val*h.cols+j.val) byte)=
          encode I S (forwardReference (algebra R .forward) h (.epsilon (EpsilonABI.convert epsilon.host)) (logicalTerms logical) i j) byte) ∧
      (∀ i : Fin h.rows,final ("r",i.val)=savedReference (algebra R .forward) h (.epsilon (EpsilonABI.convert epsilon.host)) (logicalTerms logical) i) ∧
      (∀ i : Fin h.rows,∀ byte : Fin 4,
        commit storage (encode I S) heap (fun r o => final (r,o)) (address storage "r" (by simp [buffers]) i.val byte)=
          encode I S (savedReference (algebra R .forward) h (.epsilon (EpsilonABI.convert epsilon.host)) (logicalTerms logical) i) byte) ∧
      Frame h .forward (lift storage heap decodeBytes other) final ∧
      (∀ addr,¬written storage addr → commit storage (encode I S) heap (fun r o => final (r,o)) addr=heap addr) := by
  obtain ⟨final,run,values,saved,frame⟩:=RuntimeForward.Tensor.forward (algebra R .forward) h (.epsilon epsilon.runtime) storage tensors binding heap decodeBytes other (logicalTerms logical) (word_reads _ tensors heap logical reads)
  rw [epsilon.converted] at values saved
  refine ⟨final,run,?_,saved,?_,frame,fun addr outside => commit_frame h .forward storage (encode I S) heap (fun r o => final (r,o)) addr outside⟩
  · intro i j byte
    rw [commit_output h .forward storage (encode I S) heap (fun r o => final (r,o)) "y" (by simp [outputs])
      _ (by simpa [cells] using Native.offset_bound h i.val j.val i.isLt j.isLt) byte,values]
  · intro i byte
    rw [commit_output h .forward storage (encode I S) heap (fun r o => final (r,o)) "r" (by simp [outputs])
      i.val (by simpa [cells] using i.isLt) byte,saved]

theorem native_kernel_backward (R : Reductions) (S : Selection) (I : Instructions)
    (h : Call) (storage : Placement h .backward) (tensors : RegionName → Torch.TensorMetadata)
    (binding : Binding h .backward storage tensors) (heap : Nat → Fin 256)
    (other : Cell Term) (logical : RegionName → Nat → Word) (reads : WordReadLaw .backward tensors heap logical) :
    ∃ final,FPTensor.backwardRun (algebra R .weight) (algebra R .input) h zero (lift storage heap decodeBytes other)=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,∀ byte : Fin 4,
        commit storage (encode I S) heap (fun r o => final (r,o))
          (address storage "dx" (by simp [buffers]) (i.val*h.cols+j.val) byte)=
          encode I S (inputGradientReference (algebra R .input) h (logicalTerms logical) i j) byte) ∧
      (∀ j : Fin h.cols,∀ byte : Fin 4,
        commit storage (encode I S) heap (fun r o => final (r,o))
          (address storage "dw" (by simp [buffers]) j.val byte)=
          if h.rows=0 then 0 else encode I S (weightGradientReference (algebra R .weight) h (logicalTerms logical) j) byte) ∧
      Frame h .backward (lift storage heap decodeBytes other) final ∧
      (∀ addr,¬written storage addr → commit storage (encode I S) heap (fun r o => final (r,o)) addr=heap addr) := by
  obtain ⟨final,run,dx,dw,frame⟩:=FPTensor.backward (algebra R .weight) (algebra R .input) h zero storage tensors binding heap decodeBytes other (logicalTerms logical) (word_reads _ tensors heap logical reads)
  refine ⟨final,run,?_,?_,frame,fun addr outside => commit_frame h .backward storage (encode I S) heap (fun r o => final (r,o)) addr outside⟩
  · intro i j byte
    rw [commit_output h .backward storage (encode I S) heap (fun r o => final (r,o)) "dx" (by simp [outputs])
      _ (by simpa [cells] using Native.offset_bound h i.val j.val i.isLt j.isLt) byte,dx]
  · intro j byte
    rw [commit_output h .backward storage (encode I S) heap (fun r o => final (r,o)) "dw" (by simp [outputs])
      j.val (by simpa [cells] using j.isLt) byte,dw]
    split <;> simp_all [zero_bytes]

end Native

#axiomsClean real_kernel_forward
#axiomsClean real_kernel_backward
#axiomsClean native_kernel_forward
#axiomsClean native_kernel_backward
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.TensorBridge

import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RawInputs
import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.EntryInputs

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RawNative
open VeriTile Triton FP.Structural KernelTensor KernelStorage NativeGraph FPEquiv
open FPGridForward (Heap)
noncomputable section

abbrev forwardReference := RuntimeForward.Tensor.forwardReference (α:=Term)
abbrev savedReference := RuntimeForward.Tensor.savedReference (α:=Term)
abbrev inputGradientReference := FPTensor.inputReference (α:=Term)
abbrev weightGradientReference := FPTensor.weightReference (α:=Term)

theorem kernel_forward (R : Reductions) (S : Selection) (I : Instructions)
    (h : Call) (epsilon : EpsilonABI.Binding h.eps) (initial : FPGridForward.Heap Term)
    (logical : RegionName → Nat → Term)
    (reads : ∀ role, role∈inputs .forward → ∀ off, off<cells h role →
      (initial (role,off)).read .real=logical role off) :
    ∃ final, RuntimeForward.Grid.run (algebra R .forward) h (.epsilon epsilon.runtime) initial=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,∀ byte : Fin 4,
        encode I S (final ("y",i.val*h.cols+j.val)) byte=
          encode I S (forwardReference (algebra R .forward) h (.epsilon (EpsilonABI.convert epsilon.host)) logical i j) byte) ∧
      (∀ i : Fin h.rows,final ("r",i.val)=
        savedReference (algebra R .forward) h (.epsilon (EpsilonABI.convert epsilon.host)) logical i) ∧
      Frame h .forward initial final := by
  obtain ⟨final,run,values,saved,frame⟩ := RuntimeForward.Tensor.forward_raw
    (algebra R .forward) h (.epsilon epsilon.runtime) initial logical reads
  rw [epsilon.converted] at values saved
  exact ⟨final,run,fun i j byte => congrArg (fun cell => encode I S cell byte) (values i j),saved,frame⟩

theorem kernel_backward (R : Reductions) (S : Selection) (I : Instructions)
    (h : Call) (initial : FPGridForward.Heap Term) (logical : RegionName → Nat → Term)
    (reads : ∀ role, role∈inputs .backward → ∀ off, off<cells h role →
      (initial (role,off)).read .real=logical role off) :
    ∃ final,FPTensor.backwardRun (algebra R .weight) (algebra R .input) h zero initial=some final ∧
      (∀ i : Fin h.rows,∀ j : Fin h.cols,∀ byte : Fin 4,
        encode I S (final ("dx",i.val*h.cols+j.val)) byte=
          encode I S (inputGradientReference (algebra R .input) h logical i j) byte) ∧
      (∀ j : Fin h.cols,∀ byte : Fin 4,encode I S (final ("dw",j.val)) byte=
        if h.rows=0 then 0 else encode I S (weightGradientReference (algebra R .weight) h logical j) byte) ∧
      Frame h .backward initial final := by
  obtain ⟨final,run,dx,dw,frame⟩ := FPTensor.backward_raw
    (algebra R .weight) (algebra R .input) h zero initial logical reads
  refine ⟨final,run,fun i j byte => congrArg (fun cell => encode I S cell byte) (dx i j),?_,frame⟩
  intro j byte
  rw [dw]
  split <;> simp_all [zero_bytes]

abbrev native_reduction_coverage := @NativeGraph.reduction_complete
abbrev native_weight_fusion_disabled := NativeGraph.weight_fusion_disabled
abbrev native_compiler_expression := @NativeGraph.compiled_expression
abbrev row_integer_operations := NativeGrid.row_words
abbrev weight_integer_operations := NativeGrid.weight_row_words

#axiomsClean kernel_forward
#axiomsClean kernel_backward
#specNonCircular forwardReference avoiding [FP.Structural.exec,Kernels.forwardRuntime,Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular inputGradientReference avoiding [FP.Structural.exec,Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular weightGradientReference avoiding [FP.Structural.exec,Kernels.forward,Kernels.backward,Kernels.weightBackward]
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RawNative

import bench.repairs.tritonbench_g.fast_rms_layernorm.Proofs.RawInputs

namespace VeriTile.Bench.Repairs.FastRmsLayernorm.RawCorrect
open VeriTile Triton KernelTensor KernelStorage
open RealGridForward (read Heap)
noncomputable section

abbrev output := @RealReference.output
abbrev inputGradient := @RealReference.inputGradient
abbrev weightGradient := @RealReference.weightGradient

theorem kernel_forward (h : Call) (initial : Heap)
    (logical : RegionName → Nat → ℝ)
    (reads : ∀ role, role∈inputs .forward → ∀ off, off<cells h role →
      read (initial (role,off))=logical role off) :
    ∃ final,RealGridForward.run h initial=some final ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (final ("y",row.val*h.cols+j.val))=
        RealReference.output h.gemma h.eps (rowValues h logical "x" row) (weightValues h logical) j) ∧
      (∀ row : Fin h.rows,read (final ("r",row.val))=RealReference.inv h.eps (rowValues h logical "x" row)) ∧
      Frame h .forward initial final :=
  RealTensor.forward_raw h initial logical reads

theorem kernel_backward (h : Call) (initial : Heap)
    (logical : RegionName → Nat → ℝ)
    (reads : ∀ role, role∈inputs .backward → ∀ off, off<cells h role →
      read (initial (role,off))=logical role off)
    (saved : ∀ row : Fin h.rows,logical "r" row.val=RealReference.inv h.eps (rowValues h logical "x" row)) :
    ∃ final,RealTensor.backwardRun h initial=some final ∧
      (∀ row : Fin h.rows,∀ j : Fin h.cols,read (final ("dx",row.val*h.cols+j.val))=
        RealReference.inputGradient h.gemma h.eps (rowValues h logical "x" row)
          (weightValues h logical) (rowValues h logical "dx" row) j) ∧
      (∀ j : Fin h.cols,read (final ("dw",j.val))=
        RealReference.weightGradient h.eps (rowValues h logical "x") (rowValues h logical "dx") j) ∧
      Frame h .backward initial final :=
  RealTensor.backward_raw h initial logical reads saved

abbrev input_derivative := @RealReference.full_input_directional
abbrev weight_derivative := @RealReference.full_weight_directional
abbrev row_integer_operations := NativeGrid.row_words
abbrev weight_integer_operations := NativeGrid.weight_row_words


#axiomsClean kernel_forward
#axiomsClean kernel_backward
#specNonCircular output avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular inputGradient avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
#specNonCircular weightGradient avoiding [Kernels.forward,Kernels.backward,Kernels.weightBackward]
end
end VeriTile.Bench.Repairs.FastRmsLayernorm.RawCorrect

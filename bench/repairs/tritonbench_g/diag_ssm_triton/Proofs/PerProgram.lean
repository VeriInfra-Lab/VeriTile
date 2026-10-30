import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.SourceFlatPreservation
import bench.repairs.tritonbench_g.diag_ssm_triton.Proofs.IOSafety
import bench.tritonbench_g.diag_ssm_triton.Proofs.ForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.BackwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexForwardIO
import bench.tritonbench_g.diag_ssm_triton.Proofs.ComplexBackwardIO

/-! The four existing finite flat contracts now execute the actual copied
bodies. Checked safety, supported-fragment and final-memory relations transport
the independent old results; zero length keeps its original input and alias
scope. No historical whole-state identity is assumed. -/
set_option autoImplicit false
namespace VeriTile.Bench.Repairs.DiagSsmTriton.PerProgram
open VeriTile VeriTile.Triton
open VeriTile.Bench.TritonBenchG.DiagSsmTriton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 3000000

theorem forward (p : ForwardIO.Layout) (length : Nat) :
    Spec.Real (ForwardIO.Implements (Kernels.forwardTyped p.s p.x p.l p.y
      length p.batch p.dim p.B).eraseDType p length)  := by
  intro hsep A hd hr hc u hu hb
  obtain ⟨old, he, hv, hf⟩ := ForwardIO.original_correct .fp32 p length hsep A hd hr hc u hu hb
  have sa := IOSafety.forward_trace_of_positive p length hsep u A.extent hb
  have sb := (SourceReal.forward_trace p length u A.extent).mp sa
  obtain ⟨a, b, ha, hbRun, hm⟩ := SourceReal.forward_run_related p length u
  have heRef := (ForwardProjection.typed_source_flat_exec .fp32 A p.s p.x p.l p.y
    length p.batch p.dim p.B (A.flattenState u)).symm.trans he
  obtain ⟨v, heNew, hmem⟩ := SourceFlat.transfer A hd hc _ _ u a b old hu ha hbRun hm.memory
    sa sb (SourceFlat.forward_flatten p length) (SourceFlat.reference_forward_flatten p length) heRef
  refine ⟨v, heNew, ?_, ?_⟩
  · simpa only [BlockState.readMem, hmem] using hv
  · simpa only [hmem] using hf

theorem backward (p : BackwardIO.Layout) (length : Nat) :
    Spec.Real (BackwardIO.Implements (Kernels.backwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length)  := by
  intro hsep A hd hr hc u hu hb
  obtain ⟨old, he, hx, hs, hl, hf⟩ := BackwardIO.original_correct .fp32 p length hsep A hd hr hc u hu hb
  have sep : 0 < length → RegionCorrectness.BackwardSeparation p := fun hp =>
    ⟨(hsep hp).initial, (hsep hp).output, (hsep hp).gradient, (hsep hp).transition,
      (hsep hp).initialGradient, (hsep hp).transitionGradient, (hsep hp).terminal⟩
  have sa := IOSafety.backward_trace_of_positive p length sep u A.extent hb
  have sb := (SourceBackward.backward_trace p length u A.extent).mp sa
  obtain ⟨a, b, ha, hbRun, hm⟩ := SourceBackward.backward_run_memory p length u
  have heRef := (BackwardProjection.typed_source_flat_exec .fp32 A p.s p.l p.y p.gs p.gx p.gl p.gy
    length p.batch p.dim p.B (A.flattenState u)).symm.trans he
  obtain ⟨v, heNew, hmem⟩ := SourceFlat.transfer A hd hc _ _ u a b old hu ha hbRun hm
    sa sb (SourceFlat.backward_flatten p length) (SourceFlat.reference_backward_flatten p length) heRef
  refine ⟨v, heNew, ?_, ?_, ?_, ?_⟩
  · simpa only [BlockState.readMem, hmem] using hx
  · simpa only [BlockState.readMem, hmem] using hs
  · simpa only [BlockState.readMem, hmem] using hl
  · simpa only [hmem] using hf

theorem complex_forward (p : ComplexForwardLoop.Layout) (length : Nat) :
    Spec.Real (ComplexForwardIO.Implements (Kernels.complexForwardTyped p.s p.x p.y p.l
      length p.batch p.dim p.B).eraseDType p length)  := by
  intro A hd hc u hu hb
  obtain ⟨hs, old, he, hv, hf⟩ := ComplexForwardIO.original_correct p length A hd hc u hu hb
  have sb : (Erasure.complexForward Bool.false p.s p.x p.y p.l
      length p.batch p.dim p.B).toAlgKernel.TraceSafe A.extent u := by
    simpa only [Kernel.TraceSafe, Erasure.complexForward_body] using hs
  have sa := (SourceComplex.forward_trace p length u A.extent).mpr sb
  obtain ⟨a, b, ha, hbRun, hm⟩ := SourceComplex.forward_run_related p length u
  have heRef : exec (A.flattenKernel (Erasure.complexForward Bool.false p.s p.x p.y p.l
      length p.batch p.dim p.B).toAlgKernel) (A.flattenState u) = some old := by
    simpa only [exec, FlatAlloc.flattenKernel, Erasure.complexForward_body] using he
  obtain ⟨v, heNew, hmem⟩ := SourceFlat.transfer A hd hc _ _ u a b old hu ha hbRun hm.memory
    sa sb (SourceFlat.complexForward_flatten p length) (ComplexForwardSafety.flatten_ok p length) heRef
  refine ⟨sa, v, heNew, ?_, ?_⟩
  · simpa only [ComplexForwardIO.readFlat, BlockState.readMem, hmem] using hv
  · simpa only [hmem] using hf

theorem complex_backward (p : ComplexBackwardLoop.Layout) (length : Nat) :
    Spec.Real (ComplexBackwardIO.Implements (Kernels.complexBackwardTyped p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).eraseDType p length)  := by
  intro hsep A hd hc u hu hb
  obtain ⟨hs, old, he, hx, hsOut, hl, hf⟩ := ComplexBackwardIO.original_correct p length hsep A hd hc u hu hb
  have sb : (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).toAlgKernel.TraceSafe A.extent u := by
    simpa only [Kernel.TraceSafe, Erasure.complexBackward_body] using hs
  have sa := (SourceComplexBackward.backward_trace p length u A.extent).mpr sb
  obtain ⟨a, b, ha, hbRun, hm⟩ := SourceComplexBackward.backward_run_memory p length u
  have heRef : exec (A.flattenKernel (Erasure.complexBackward Bool.false p.s p.l p.y p.gs p.gx p.gl p.gy
      length p.batch p.dim p.B).toAlgKernel) (A.flattenState u) = some old := by
    simpa only [exec, FlatAlloc.flattenKernel, Erasure.complexBackward_body] using he
  obtain ⟨v, heNew, hmem⟩ := SourceFlat.transfer A hd hc _ _ u a b old hu ha hbRun hm
    sa sb (SourceFlat.complexBackward_flatten p length) (ComplexBackwardSafety.flatten_ok p length) heRef
  refine ⟨sa, v, heNew, ?_, ?_, ?_, ?_⟩
  · simpa only [ComplexBackwardIO.readFlat, BlockState.readMem, hmem] using hx
  · simpa only [ComplexBackwardIO.readFlat, BlockState.readMem, hmem] using hsOut
  · simpa only [ComplexBackwardIO.readFlat, BlockState.readMem, hmem] using hl
  · simpa only [hmem] using hf

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DiagSsmTriton.PerProgram

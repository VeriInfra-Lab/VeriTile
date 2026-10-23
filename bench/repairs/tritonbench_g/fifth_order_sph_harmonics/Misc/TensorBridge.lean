import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Misc.KernelRealResults
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Misc.KernelNativeResults
import bench.repairs.tritonbench_g.fifth_order_sph_harmonics.Proofs.KernelEntries

/-! Optional actual Tensor byte readback and native address-binding exports.
Primary forward/backward mathematical and native kernel results stay in their entries. -/
namespace VeriTile.Bench.Repairs.FifthOrderSphHarmonics.TensorBridge
open VeriTile.Triton FP.Structural KernelEntries
noncomputable section
set_option maxHeartbeats 5000000

/-- Complete forward source execution, all eleven original exact-decimal
polynomials on actual Tensor inputs, and every untouched tagged memory cell. -/
theorem real_forward (f : Forward) (memory : Nat → ℝ) :
    ∃ final,Grid.launch KernelScalar.realBackend .forward f.params
      (KernelScalar.realInitial f.params f.buffers memory)=some final ∧
      (∀ r,r<f.params.rows → ∀ c : Fin 11,
        tensorCell final outputRegion f.output r (f.params.columnOffset+c.val)=
          .mk .real (realForward f memory r c)) ∧
      KernelScalar.Frame .forward f.params (KernelScalar.realInitial f.params f.buffers memory) final := by
  obtain ⟨final,run,values,frame⟩:=KernelRealResults.forward f.params f.buffers memory (fit f)
  refine ⟨final,run,?_,frame⟩
  intro r hr c
  rw [tensorCell,output_index f r hr c]
  exact (values r hr c).trans (congrArg (Cell.mk .real) (congrFun (real_forward_reference f memory r hr) c))

/-- Complete backward source execution equals the true Jacobian transpose of
those eleven polynomials. Only active prepared gradient entries are zero;
all padding and untouched memory retain their actual initial values. -/
theorem real_backward (f : Forward) (w : Backward f) (memory : Nat → ℝ) (zero : RealPrepared w memory) :
    ∃ final,Grid.launch KernelScalar.realBackend .backward w.params
      (KernelScalar.realInitial w.params w.buffers memory)=some final ∧
      (∀ r,r<f.params.rows → ∀ c : Fin 3,tensorCell final gradientRegion w.gradient r c.val=
        .mk .real (realBackward w memory r c)) ∧
      KernelScalar.Frame .backward w.params (KernelScalar.realInitial w.params w.buffers memory) final := by
  obtain ⟨final,run,values,frame⟩:=KernelRealResults.backward w.params w.buffers memory (fit f)
    (real_zero w memory zero)
  refine ⟨final,run,?_,frame⟩
  intro r hr c
  rw [tensorCell,gradient_index w r hr c]
  exact (values r hr c).trans (congrArg (Cell.mk .real) (congrFun (real_backward_reference w memory r hr) c))

/-- Native-word source execution uses concrete separate-rounding FP32
semantics; no local numerical oracle or output-shaped premise is assumed. -/
theorem native_forward (f : Forward) (heap : Storage.Heap) :
    ∃ final,Grid.launch (KernelScalar.forwardBackend KernelScalar.selectedProfile) .forward f.params
      (Physical.backed .forward f.params f.buffers heap)=some final ∧
      (∀ r,r<f.params.rows → ∀ c : Fin 11,
        tensorCell final outputRegion f.output r (f.params.columnOffset+c.val)=
          .mk .real (Packing.word (nativeForward f heap r c))) ∧
      KernelScalar.Frame .forward f.params (Physical.backed .forward f.params f.buffers heap) final := by
  obtain ⟨final,run,values,frame⟩:=KernelNativeResults.forward f.params f.buffers heap (fit f)
  refine ⟨final,run,?_,frame⟩
  intro r hr c
  rw [tensorCell,output_index f r hr c]
  exact (values r hr c).trans (congrArg (fun v=>Cell.mk .real (Packing.word v))
    (congrFun (native_forward_reference f heap r hr) c))

/-- The actual prepared initial +0 is read and added in source order. The
upstream logical reader permits every admitted nonnegative queried stride. -/
theorem native_backward (f : Forward) (w : Backward f) (heap : Storage.Heap) (zero : NativePrepared w heap) :
    ∃ final,Grid.launch (KernelScalar.backwardBackend KernelScalar.selectedProfile) .backward w.params
      (Physical.backed .backward w.params w.buffers heap)=some final ∧
      (∀ r,r<f.params.rows → ∀ c : Fin 3,tensorCell final gradientRegion w.gradient r c.val=
        .mk .real (Packing.word (nativeBackward w heap r c))) ∧
      KernelScalar.Frame .backward w.params (Physical.backed .backward w.params w.buffers heap) final := by
  obtain ⟨final,run,values,frame⟩:=KernelNativeResults.backward w.params w.buffers heap (fit f)
  refine ⟨final,run,?_,frame⟩
  intro r hr c
  rw [tensorCell,gradient_index w r hr c]
  exact (values r hr c).trans (congrArg (fun v=>Cell.mk .real (Packing.word v))
    (congrFun (native_backward_reference w heap zero r hr) c))

#axiomsClean real_forward
#axiomsClean real_backward
#axiomsClean native_forward
#axiomsClean native_backward

/-- Every selected output byte at its actual Tensor address and the full
exterior byte frame follow from the same source store replay. -/
theorem forward_bytes (f : Forward) (heap : Storage.Heap) :
    let result:=(Physical.launch (KernelScalar.forwardBackend KernelScalar.selectedProfile)
      .forward f.params f.buffers heap).bytes
    (∀ r,r<f.params.rows → ∀ c : Fin 11,∀ k : Fin 4,result
      (f.output.tensor.storage.address (f.output.tensor.physicalIndex
        (r*(f.request.output f.allocated).metadata.columns+(f.params.columnOffset+c.val)))+k.val)=
      Packing.word (nativeForward f heap r c) k) ∧
    (∀ address,(∀ r,r<f.params.rows → ∀ c : Fin 11,∀ k : Fin 4,
      address≠f.buffers.destination+4*(r*f.params.outputWidth+f.params.columnOffset+c.val)+k.val) →
      result address=heap address) := by
  have h:=Equivalence.forward (KernelScalar.forwardBackend KernelScalar.selectedProfile)
    KernelScalar.wordBase KernelScalar.valueBase KernelScalar.selectedProfile f.params f.buffers heap (fit f)
    (fun _ _ c=>KernelScalar.forward_laws _ _ c)
  refine ⟨?_,h.2⟩
  intro r hr c k
  rw [←output_address f r hr c]
  exact (h.1 r hr c k).trans (congrFun (congrArg Packing.word
    (congrFun (native_forward_reference f heap r hr) c)) k)

theorem backward_bytes (f : Forward) (w : Backward f) (heap : Storage.Heap) (zero : NativePrepared w heap) :
    let result:=(Physical.launch (KernelScalar.backwardBackend KernelScalar.selectedProfile)
      .backward w.params w.buffers heap).bytes
    (∀ r,r<f.params.rows → ∀ c : Fin 3,∀ k : Fin 4,result
      (w.gradient.tensor.storage.address (w.gradient.tensor.physicalIndex
        (r*w.gradientObject.metadata.columns+c.val))+k.val)=Packing.word (nativeBackward w heap r c) k) ∧
    (∀ address,(∀ r,r<f.params.rows → ∀ c : Fin 3,∀ k : Fin 4,
      address≠w.buffers.destination+4*(r*3+c.val)+k.val) → result address=heap address) := by
  have h:=Equivalence.backward (KernelScalar.backwardBackend KernelScalar.selectedProfile)
    KernelScalar.wordBase KernelScalar.valueBase KernelScalar.selectedProfile w.params w.buffers heap (fit f)
    (fun _ _ c=>KernelScalar.backward_laws _ _ c)
  refine ⟨?_,h.2⟩
  intro r hr c k
  rw [←gradient_address w r hr c]
  exact (h.1 r hr c k).trans (congrFun (congrArg Packing.word
    (congrFun (native_backward_reference w heap zero r hr) c)) k)

export Lowering (forward_masks forward_pointer coordinate_pointer upstream_pointer gradient_pointer)

#axiomsClean forward_bytes
#axiomsClean backward_bytes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.FifthOrderSphHarmonics.TensorBridge

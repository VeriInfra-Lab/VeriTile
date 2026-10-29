import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelRealResult
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelNativeResult
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelInputs

namespace VeriTile.Bench.Repairs.Conv2d.TensorBridge
open VeriTile.Triton FP.Structural KernelTensor KernelEntries KernelNumeric
noncomputable section

/-- Kernel-only exact grouped cross-correlation. The original full grid runs
the repaired readable shared DSL; output observations use the actual bound
Tensor and initial caller storage. No host execution or result oracle is a premise. -/
theorem correct_real (p : Prepared) (heap : Nat → ℝ) :
    SourceSafe p.call.shape (inputWords p.call.shape) (weightWords p.call.shape) ∧
    ∃ final,KernelGrid.launch (algebra KernelReal.operations) (halfDot KernelReal.operations)
      (defaultDot KernelReal.operations) p.call.shape p.call.flags (initial p heap)=some final ∧
      (∀ x,p.call.shape.Active x → realResult p heap final x=realTarget p heap x) ∧
      (∀ (r : RegionName) a,r≠"output" ∨ p.call.shape.size≤a → final.mem r a=(initial p heap).mem r a) ∧
      (∀ q,q<p.output.pointer ∨ p.output.pointer+4*p.call.shape.size≤q → realHeap p heap final q=heap q) := by
  obtain ⟨final,he,hr,hf⟩:=KernelEntries.real_correct p heap
  exact ⟨domain_safety p.call p.domain,final,he,hr,hf,real_frame p heap final⟩

/-- Kernel-only native equivalence on all raw FP32 input words. The common
readable source runs over the complete original grid. The independent graph
expands each K32 dot into literal scalar/K8/K16 instructions, with the observed
carry absorption explicit. Only scalar/conversion/one-instruction laws remain. -/
theorem fp_equivalent (p : Prepared) (heap : KernelStorage.Heap)
    (actual numerics : KernelNative.Instructions)
    (laws : KernelNative.LocalLaws actual (KernelNative.software numerics)) :
    let o:=KernelNative.operations actual p.call.flags
    ∃ final,KernelGrid.launch (algebra o) (halfDot o) (defaultDot o) p.call.shape p.call.flags
      (initial p (KernelStorage.read heap))=some final ∧
      (∀ x,p.call.shape.Active x → nativeResult actual p heap final x=
        Software.value (nativeTarget (KernelNative.software numerics) p heap x)) ∧
      (∀ x,p.call.shape.Active x → ∀ k : Fin 4,
        nativeHeap actual p heap final (p.output.pointer+4*outputPointer p.call.shape x+k.val)=
          nativeTarget (KernelNative.software numerics) p heap x k) ∧
      (∀ q,q<p.output.pointer ∨ p.output.pointer+4*p.call.shape.size≤q → nativeHeap actual p heap final q=heap q) ∧
      (∀ q,q<p.call.input.capacity → nativeHeap actual p heap final (p.call.input.storageBase+q)=heap (p.call.input.storageBase+q)) ∧
      (∀ q,q<p.call.weight.capacity → nativeHeap actual p heap final (p.call.weight.storageBase+q)=heap (p.call.weight.storageBase+q)) := by
  rw [KernelNative.instructions_eq actual _ laws]
  exact KernelEntries.native_correct (KernelNative.software numerics) p heap

/-- Production defaults close the numerical target to the IEEE RN-even FP32
FMA circuit. No conversion or tensor-instruction equation is required. -/
theorem fp_default (p : Prepared) (defaults : p.call.flags=⟨Bool.false,Bool.false⟩)
    (heap : KernelStorage.Heap) (actual : KernelNative.Instructions)
    (fmaLaw : ∀ a b c,actual.fmaRN a b c=Software.fmaRN a b c) :
    let o:=KernelNative.operations actual p.call.flags
    ∃ final,KernelGrid.launch (algebra o) (halfDot o) (defaultDot o) p.call.shape p.call.flags
      (initial p (KernelStorage.read heap))=some final ∧
      (∀ x,p.call.shape.Active x → nativeResult actual p heap final x=
        Software.value (KernelNative.defaultReference p.call.shape (physicalMemory p (KernelStorage.read heap)) x)) ∧
      (∀ x,p.call.shape.Active x → ∀ k : Fin 4,
        nativeHeap actual p heap final (p.output.pointer+4*outputPointer p.call.shape x+k.val)=
          KernelNative.defaultReference p.call.shape (physicalMemory p (KernelStorage.read heap)) x k) ∧
      (∀ q,q<p.output.pointer ∨ p.output.pointer+4*p.call.shape.size≤q → nativeHeap actual p heap final q=heap q) ∧
      (∀ q,q<p.call.input.capacity → nativeHeap actual p heap final (p.call.input.storageBase+q)=heap (p.call.input.storageBase+q)) ∧
      (∀ q,q<p.call.weight.capacity → nativeHeap actual p heap final (p.call.weight.storageBase+q)=heap (p.call.weight.storageBase+q)) := by
  obtain ⟨final,he,hr,hb,hf,hi,hw⟩:=KernelEntries.native_correct actual p heap
  have target (x : Conv2d.Cell) : nativeTarget actual p heap x=
      KernelNative.defaultReference p.call.shape (physicalMemory p (KernelStorage.read heap)) x := by
    unfold nativeTarget
    rw [defaults]
    exact KernelNative.default_reference actual fmaLaw _ _ _
  exact ⟨final,he,fun x hx=>(hr x hx).trans (congrArg Software.value (target x)),
    fun x hx k=>(hb x hx k).trans (congrFun (target x) k),hf,hi,hw⟩

/-- The kernel result is the prepared actual output Tensor with the allocation
identity, device and shape supplied by the existing successful allocation law. -/
theorem output_tensor (p : Prepared) :
    p.result.tensor.metadata.dtype=.floating .fp32 ∧
    p.result.tensor.metadata.shape=[p.call.shape.n,p.call.shape.outputs,p.call.shape.oh,p.call.shape.ow] ∧
    p.result.tensor.metadata.device=p.input.tensor.metadata.device ∧
    p.result.tensor.object≠p.input.tensor.object ∧ p.result.tensor.object≠p.weight.tensor.object :=
  tensor_metadata p

#axiomsClean correct_real
#axiomsClean fp_equivalent
#axiomsClean fp_default
#axiomsClean output_tensor
end
end VeriTile.Bench.Repairs.Conv2d.TensorBridge

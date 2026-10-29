import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelEntries
namespace VeriTile.Bench.Repairs.Conv2d.KernelEntries
open VeriTile.Triton FP.Structural KernelTensor
noncomputable section

/-- Caller input reads are reads of the actual bound Tensor, including its
storage offset and arbitrary positive strides. -/
theorem real_input (p : Prepared) (heap : Nat → ℝ) (n c y x : Nat)
    (hn : n<p.call.shape.n) (hc : c<p.call.shape.channels)
    (hy : y<p.call.shape.h) (hx : x<p.call.shape.w) :
    physicalMemory p heap .input ((inputView p.call.shape).offset n c y x)=
      KernelTensor.realPoint heap p.input n c y x :=
  KernelTensor.real_read heap p.input n c y x hn hc hy hx

theorem real_weight (p : Prepared) (heap : Nat → ℝ) (n c y x : Nat)
    (hn : n<p.call.shape.outputs) (hc : c<p.call.shape.cg)
    (hy : y<p.call.shape.kh) (hx : x<p.call.shape.kw) :
    physicalMemory p heap .weight ((weightView p.call.shape).offset n c y x)=
      KernelTensor.realPoint heap p.weight n c y x :=
  KernelTensor.real_read heap p.weight n c y x hn hc hy hx

theorem native_input (p : Prepared) (heap : KernelStorage.Heap) (n c y x : Nat)
    (hn : n<p.call.shape.n) (hc : c<p.call.shape.channels)
    (hy : y<p.call.shape.h) (hx : x<p.call.shape.w) :
    Software.value (physicalMemory p (KernelStorage.read heap) .input ((inputView p.call.shape).offset n c y x))=
      KernelTensor.nativePoint (fun a=>Software.value (KernelStorage.read heap a)) p.input n c y x :=
  KernelTensor.native_read (fun a=>Software.value (KernelStorage.read heap a)) p.input p.domain.inputKind n c y x hn hc hy hx

theorem native_weight (p : Prepared) (heap : KernelStorage.Heap) (n c y x : Nat)
    (hn : n<p.call.shape.outputs) (hc : c<p.call.shape.cg)
    (hy : y<p.call.shape.kh) (hx : x<p.call.shape.kw) :
    Software.value (physicalMemory p (KernelStorage.read heap) .weight ((weightView p.call.shape).offset n c y x))=
      KernelTensor.nativePoint (fun a=>Software.value (KernelStorage.read heap a)) p.weight n c y x :=
  KernelTensor.native_read (fun a=>Software.value (KernelStorage.read heap a)) p.weight p.domain.weightKind n c y x hn hc hy hx

theorem tensor_metadata (p : Prepared) :
    p.result.tensor.metadata.dtype=.floating .fp32 ∧
    p.result.tensor.metadata.shape=[p.call.shape.n,p.call.shape.outputs,p.call.shape.oh,p.call.shape.ow] ∧
    p.result.tensor.metadata.device=p.input.tensor.metadata.device ∧
    p.result.tensor.object≠p.input.tensor.object ∧ p.result.tensor.object≠p.weight.tensor.object := by
  rw [p.result.metadata,p.input.metadata,p.result.object,p.input.object,p.weight.object]
  exact ⟨p.allocation.kind.2.1,p.allocation.shape,p.allocation.device,
    p.allocation.freshInput,p.allocation.freshWeight⟩

theorem real_frame (p : Prepared) (before : Nat → ℝ) (final : State (KernelNumeric.Value ℝ))
    (q : Nat) (hq : q<p.output.pointer ∨ p.output.pointer+4*p.call.shape.size≤q) :
    realHeap p before final q=before q := by
  simp [realHeap,show ¬(p.output.pointer≤q ∧ q<p.output.pointer+4*p.call.shape.size) by omega]

#axiomsClean real_input
#axiomsClean real_weight
#axiomsClean native_input
#axiomsClean native_weight
#axiomsClean tensor_metadata
#axiomsClean real_frame
end
end VeriTile.Bench.Repairs.Conv2d.KernelEntries

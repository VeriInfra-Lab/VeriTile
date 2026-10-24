import bench.repairs.tritonbench_g.matmul_dequantize_int4.Correct
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.KernelTensor
import bench.repairs.tritonbench_g.matmul_dequantize_int4.Proofs.QuantRealTensor

/-! Optional checked Tensor invocation applications for the matrix product. -/
namespace VeriTile.Bench.Repairs.MatmulDequantizeInt4.TensorBridge
open Checked
noncomputable section

namespace Correct
/-- Kernel-only public specification: the selected shared DSL executes on the
actual Tensor views and yields the independent ideal-real matrix product.
All machine bounds are derived from the unchanged public checks. -/
theorem kernel_correct (p : Metadata) (t : TensorBinding.Tensors p) (h : InputContract p)
    (i : Fin 17) (raw : TensorBinding.Heap) (heap : RealTensor.RealHeap) :
    KernelTensor.MachineSafe p i ∧
    (observations p).Pairwise (fun x y => x.storageObject≠y.storageObject) ∧
    VeriTile.Spec.Real (∃ state,KernelSource.run KernelReal.scalar KernelReal.dot p.call i
        (RealTensor.entry p raw heap)=some state ∧
      KernelTensor.RealPost p t i raw heap (RealTensor.physical p heap state)) :=
  ⟨KernelTensor.machine_safe p h i,TensorBinding.allocation_names p t h,
    KernelTensor.real p t h i raw heap⟩

theorem quantize (p : QuantMetadata) (t : QuantTensor.Tensors p) (raw : QuantTensor.Heap)
    (heap : QuantRealTensor.Heap) (h : QuantInputContract p (QuantTensor.memory p t raw))
    (tree : ReductionBinding p.call) :
    ∃ result,QuantRealTensor.invoke p t raw heap=.ok result ∧ QuantRealTensor.Guarantee p t heap result :=
  QuantRealTensor.whole_tensor p t raw heap h tree


theorem tensor (p : Metadata) (t : TensorBinding.Tensors p) (h : InputContract p)
    (i : Fin 17) (raw : TensorBinding.Heap) (heap : RealTensor.RealHeap) :
    ∃ result,RealTensor.invoke p t i raw heap=.ok (some result) ∧ RealTensor.Guarantee p t i raw heap result :=
  RealTensor.whole_tensor p t h i raw heap

#axiomsClean tensor
end Correct

namespace FPEquiv
/-- Kernel-only native specification. Ordinary local instruction/compiler
bindings transport the observed fused graph; no whole-output premise occurs. -/
theorem kernel_equiv (p : Metadata) (t : TensorBinding.Tensors p) (h : InputContract p)
    (actual : Native.Instructions) (mma : Opcode → ISA.MMA16) (ports : Native.Obligations actual mma)
    (i : Fin 17) (heap : TensorBinding.Heap) :
    KernelTensor.MachineSafe p i ∧
    (observations p).Pairwise (fun x y => x.storageObject≠y.storageObject) ∧
    ∃ state,Native.grid actual p.call i (Physical.load (TensorBinding.memory p heap))=some state ∧
      KernelTensor.NativePost p t mma i heap
        (TensorBinding.physical p heap (Physical.commit (TensorBinding.memory p heap) state)) :=
  ⟨KernelTensor.machine_safe p h i,TensorBinding.allocation_names p t h,
    KernelTensor.native p t h actual mma ports i heap⟩

theorem quantize (p : QuantMetadata) (t : QuantTensor.Tensors p) (heap : QuantTensor.Heap)
    (h : QuantInputContract p (QuantTensor.memory p t heap)) (tree : ReductionBinding p.call)
    (actual : QuantFP.Instructions) (ports : QuantFP.Obligations actual) :
    ∃ result,QuantTensor.invoke p t actual heap=.ok result ∧ QuantTensor.Guarantee p t heap result :=
  QuantTensor.whole_tensor p t heap h tree actual ports


theorem tensor (p : Metadata) (t : TensorBinding.Tensors p) (h : InputContract p)
    (actual : Native.Instructions) (mma : Opcode → ISA.MMA16) (ports : Native.Obligations actual mma)
    (i : Fin 17) (heap : TensorBinding.Heap) :
    ∃ result,TensorBinding.invoke p t actual i heap=.ok (some result) ∧
      TensorBinding.Guarantee p t mma i heap result :=
  TensorBinding.whole_tensor p t h actual mma ports i heap

#axiomsClean tensor
end FPEquiv
end
end VeriTile.Bench.Repairs.MatmulDequantizeInt4.TensorBridge

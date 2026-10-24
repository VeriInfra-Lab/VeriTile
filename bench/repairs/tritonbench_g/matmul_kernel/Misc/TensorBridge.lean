import bench.repairs.tritonbench_g.matmul_kernel.Correct
import bench.repairs.tritonbench_g.matmul_kernel.Misc.NativeCorrect
import bench.repairs.tritonbench_g.matmul_kernel.Proofs.KernelReplay
namespace VeriTile.Bench.Repairs.MatmulKernel.TensorBridge
open VeriTile.Triton FP.Structural KernelTensor KernelEntries
noncomputable section
/-- Complete shared-DSL kernel, ideal Real matrix product on actual shared
Tensor inputs, actual destination readback and the full tagged memory frame.
The prepared arguments contain input metadata and ordinary Tensor API laws. -/
theorem correct (p : Prepared) (memory : Nat → ℝ) :
    ∃ final,KernelSource.launch KernelReal.scalar (KernelScalar.dotAcc KernelReal.dot) p.config (initial p memory)=some final ∧
      (∀ w,p.config.Active w → realResult p final w.1 w.2=realTarget p memory w.1 w.2) ∧
      (∀ (region : RegionName) offset,(region≠"C" ∨ ∀ w,p.config.Active w → offset≠p.config.C.addr w.1 w.2) →
        final.mem region offset=(initial p memory).mem region offset) :=
  KernelEntries.real_correct p memory
#axiomsClean correct
end
end VeriTile.Bench.Repairs.MatmulKernel.TensorBridge

namespace VeriTile.Bench.Repairs.MatmulKernel.TensorBridge
open VeriTile.Triton FP.Structural KernelTensor KernelEntries
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩
/-- Full shared source equivalence to the independent source-ordered K8 native
model. Fixed MMA/WGMMA primitive realization is explicit; no whole-dot oracle,
output equality, conversion callback, or finite-value premise is supplied. -/
theorem fp_equiv (instructions : KernelNative.Instructions) (p : Prepared) (before : KernelStorage.Heap) :
    ∃ final,KernelSource.launch KernelNative.scalar (KernelScalar.dotAcc (KernelNative.dot instructions p.config.opcode))
      p.config (initial p (KernelStorage.read before))=some final ∧
      KernelReplay.replay KernelNative.scalar (KernelScalar.dotAcc (KernelNative.dot instructions p.config.opcode)) p
        (programs p.config) (initial p (KernelStorage.read before)) before=some (final,nativeHeap p before final) ∧
      (∀ w,p.config.Active w → nativeResult p before final w.1 w.2=nativeTarget instructions p before w.1 w.2) ∧
      (∀ w,p.config.Active w → ∀ byte : Fin 4,nativeHeap p before final
        (p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val)=
          KernelStorage.bytes (nativeTarget instructions p before w.1 w.2) byte) ∧
      (∀ q,(∀ w,p.config.Active w → ∀ byte : Fin 4,
        q≠p.c.tensor.storage.address (p.c.tensor.physicalIndex (w.1*p.config.N+w.2))+byte.val) →
        nativeHeap p before final q=before q) :=
  by
    obtain ⟨final,he,hr,hb,hf⟩:=KernelResult.native instructions p before
    exact ⟨final,he,KernelReplay.native_replay instructions p before final he,hr,hb,hf⟩
#axiomsClean fp_equiv
end
end VeriTile.Bench.Repairs.MatmulKernel.TensorBridge

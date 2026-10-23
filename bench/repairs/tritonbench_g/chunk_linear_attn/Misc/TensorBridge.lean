import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.RealKernelResult
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.NativeKernelResult
import bench.repairs.tritonbench_g.chunk_linear_attn.Misc.LegacyCorrect
import bench.repairs.tritonbench_g.chunk_linear_attn.Misc.LegacyFPEquiv
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelResults

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.TensorBridge
open VeriTile Triton FP.Structural
open KernelPhysical KernelResults
noncomputable section

/-- All four shared source kernels, the independent causal formula/VJP, actual
Tensor outputs, and complete input-storage/exterior frame. The wrapper supplies
the public guards; setup and allocation are ordinary supplied buffer bindings.
Execution success is a conclusion. No host/autograd implementation is proved. -/
theorem kernel_correct (c : Host.Invocation) (flags : Admission.RuntimeFlags)
    (h : Contract.InputContract c flags) (a : Host.Allocations) (resources : Resources c a)
    (before : Nat → Cell ℝ) (fallback : Memory ℝ)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.mathCall work) :
    Spec.Real (∃ final,runRealFor c a work before fallback=some final ∧
      AddressConformance c a ∧ RealTensorResult c a resources before fallback final.2) := by
  have valid := (Contract.admitted c flags h resources.api).valid
  rw [real_order c a work legal]
  exact ⟨_,real_run c a before fallback,address_conformance c flags h a resources,
    real_tensor c a resources valid before fallback⟩

#axiomsClean kernel_correct
#axiomsClean Contract.check_iff
#axiomsClean Contract.pointers_iff
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.TensorBridge

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.TensorBridge
open VeriTile Triton FP.Structural
open KernelPhysical KernelResults
noncomputable section

/-- Exact native interpretation of the shared source under the retained fixed
FP32/FMA profile. Both sides use the same scalar ISA interpretation; there is
no numerical rewrite premise. The independent target is a coordinate-table
recurrence, connected to actual Tensor values, all output bytes and the frame.
Compiler/primitive interpretation is a model boundary, not an experiment TODO. -/
theorem kernel_native (profile : NativeTerms.FP32Profile) (c : Host.Invocation)
    (flags : Admission.RuntimeFlags) (h : Contract.InputContract c flags)
    (a : Host.Allocations) (resources : Resources c a)
    (before : BM.Read) (fallback : RegionName → Nat → BitVec 32)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.mathCall work) :
    ∃ final,runNativeFor profile c a work before fallback=some final ∧
      AddressConformance c a ∧ NativeTensorResult profile c a resources (Contract.admitted c flags h resources.api).valid
        before fallback final.2 := by
  have valid := (Contract.admitted c flags h resources.api).valid
  rw [native_order profile c a work legal]
  exact ⟨_,native_run profile c a valid before fallback,address_conformance c flags h a resources,
    native_tensor profile c a resources valid before fallback⟩

#axiomsClean kernel_native
#axiomsClean KernelTensor.native_input
#specNonCircular KernelExecution.nativeTarget avoiding [KernelExecution.run,KernelExecution.runNative,
  FP.Structural.exec,runGrid,executeFour,gridMemory,stageMemory,modelFour,completeMemory,
  NativeTerms.rawMemory,NativeBound.rawMemory]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.TensorBridge

import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.NativeGraph
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.TargetBridge
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.NativeKernelResult
open VeriTile.Triton FP.Structural NativeGraph
noncomputable section

/-- Complete source programs and their memory frames, source-derived atomic
transitions at arbitrary current outputs, and the exact independent native
64-event reference under the unchanged local compiler/scalar model. -/
theorem kernel_native (plans : ScalarTree.Plans) (S : Selection) (I : Instructions)
    (binding : CompilerBinding plans S) (h : Call) (data : FPValues.Data NG.Term)
    (valid : ∀ b,b<h.batches → data.indices b=-1 ∨ 0≤data.indices b)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (run : FPWorkset.Run algebra (primitives plans) h orders data) :
    KernelInterface.Programs algebra (primitives plans) h data ∧
      FPWorkset.SourceWorkset algebra (primitives plans) h data ∧
      ∀ (b : Fin h.batches) (n : Fin h.N),
        (binding.program (run.output b n)).eval I=
          eval I S (Target.target algebra (primitives plans) h orders data b n) := by
  refine ⟨KernelInterface.programs algebra (primitives plans) h data valid,
    FPWorkset.source_workset algebra (primitives plans) h data valid,?_⟩
  intro b n
  rw [compiled_expression,FPWorkset.readback,Target.agrees]

#axiomsClean kernel_native
#specNonCircular Target.target avoiding [FPWorkset.SourceWorkset,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.NativeKernelResult

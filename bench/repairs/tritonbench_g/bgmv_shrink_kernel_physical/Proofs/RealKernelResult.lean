import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.Real
import bench.repairs.tritonbench_g.bgmv_shrink_kernel_physical.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RealKernelResult
open VeriTile.Triton
noncomputable section

/-- Every complete source program succeeds and is framed. Its local atomic
transitions realize all 64 split events and give the independent dense Real
GEMV under ordinary per-address atomic linearization. -/
theorem kernel_real (h : Call) (u : BlockState) (indices : Nat → Int)
    (valid : ∀ b,b<h.batches → indices b=-1 ∨ 0 ≤ indices b)
    (orders : VeriTile.Bench.Repairs.BgmvShrinkKernel.Direct.Workset.Orders h)
    (run : Real.Run h orders u indices) :
    Spec.Real (KernelInterface.Programs Real.algebra Real.primitives h (Real.data h u indices) ∧
      FPWorkset.SourceWorkset Real.algebra Real.primitives h (Real.data h u indices) ∧
      ∀ (b : Fin h.batches) (n : Fin h.N),run.output b n=Target.real h u indices b n) :=
  ⟨KernelInterface.programs Real.algebra Real.primitives h (Real.data h u indices) valid,
   FPWorkset.source_workset Real.algebra Real.primitives h (Real.data h u indices) valid,
   fun b n => Real.readback h orders u indices run b n⟩

#axiomsClean kernel_real
#specNonCircular Target.real avoiding [Real.Run,Surface.source,Kernels.source,Kernels.originalFP16,Kernels.originalBF16,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernelPhysical.RealKernelResult

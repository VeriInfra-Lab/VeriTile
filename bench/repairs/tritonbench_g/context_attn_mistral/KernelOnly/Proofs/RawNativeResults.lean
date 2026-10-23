import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Kernels
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Footprint
import bench.repairs.tritonbench_g.context_attn_mistral.KernelOnly.Native

namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnMistral
open VeriTile.Bench.Repairs.ContextAttnFwd.NativeGraph (Word ScalarOps)
noncomputable section

/-- Complete raw source grid, independent recurrence equality, native scalar
observations and the exact exterior-cell frame. Integer metadata is read from
raw state; no Tensor or wrapper-check application is assumed. -/
theorem raw_kernel_native (c : Host.Call) (jobs : List (GridIndex c.grid))
    (before : State NativePhysical.Expr) (metadata : KernelFootprint.Metadata c before)
    (separate : c.regions.out≠c.regions.start ∧ c.regions.out≠c.regions.sequence)
    (actual reference : ScalarOps) (laws : NativeInstructions.ScalarLaws actual reference) :
    ∃ final,KernelReference.runGrid c (KernelNative.backend c.layout.D c.layout.window.toNat) jobs before=some final ∧
      final.mem=(KernelReference.grid c (KernelNative.backend c.layout.D c.layout.window.toNat) jobs before).mem ∧
      KernelFootprint.Metadata c final ∧
      (∀ rg off owner,KernelNative.evaluate actual c.layout.D c.layout.window.toNat
        ((final.mem rg off).read .real) owner=
        KernelNative.evaluate reference c.layout.D c.layout.window.toNat
          (((KernelReference.grid c (KernelNative.backend c.layout.D c.layout.window.toNat) jobs before).mem rg off).read .real) owner) ∧
      (∀ rg off,¬KernelFootprint.writes c jobs rg off → final.mem rg off=before.mem rg off) := by
  obtain ⟨final,execution,memory⟩ := KernelReference.whole c
    (KernelNative.backend c.layout.D c.layout.window.toNat) jobs before
  obtain ⟨framed,framedRun,finalMetadata,frame⟩ := KernelFootprint.whole_frame c
    (KernelNative.backend c.layout.D c.layout.window.toNat) jobs before metadata separate
  have same : framed=final := by rw [execution] at framedRun; exact Option.some.inj framedRun.symm
  subst framed
  refine ⟨final,execution,memory,finalMetadata,?_,frame⟩
  have equal := laws.operations_eq actual reference
  subst reference
  intro rg off owner
  rw [memory]

#axiomsClean raw_kernel_native
#specNonCircular KernelReference.grid avoiding [FP.Structural.exec,FP.Structural.run,KernelReference.runGrid]
end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel

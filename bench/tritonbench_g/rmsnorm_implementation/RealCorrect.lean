import bench.tritonbench_g.rmsnorm_implementation.Proofs.LogicalReader

/-! Kernel-only finite row specifications. The shared IO judgment includes actual
source access safety, success of both full passes, all active outputs and the
exact exterior frame. Width, epsilon and leading/input strides remain arbitrary;
only active output coordinates must be unique. IO.lean exposes the source,
channels, windows, masks, precision projection and independent RMS formula. -/
namespace VeriTile.Bench.Repairs.RmsnormImplementation.Correct
open VeriTile Triton Interface
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

specification rmsnorm_kernel_correctness (sxb sxm sxk sw sob som sok n : Nat) (eps : ℝ)
    (unique : OutputUnique n sok) :
    Spec.Real (logicalIO Bool.false sxb sxm sxk sw sob som sok n eps ⊨
      fun values k =>
        let xs := values.1
        let weights := values.2
        (xs k / Real.sqrt ((∑ j : Fin n, (xs j)^2)/(n:ℝ)+eps)) * weights k) :=
  (logical_iff Bool.false sxb sxm sxk sw sob som sok n eps).mpr
    (implements Bool.false sxb sxm sxk sw sob som sok n eps unique)

#specNonCircular formula avoiding [source,rowIO,kernel,indexCommutedKernel]
#axiomsClean rmsnorm_kernel_correctness
#stmtSurfaceSubset rmsnorm_kernel_correctness ⊆
  [Spec.Real,logicalIO,OutputUnique,LogicalScheduledKernelIO.Implements]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormImplementation.Correct

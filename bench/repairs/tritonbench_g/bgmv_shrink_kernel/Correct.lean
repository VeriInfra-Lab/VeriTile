import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Kernels
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.CheckedReal
import bench.repairs.tritonbench_g.bgmv_shrink_kernel.Proofs.Direct.Bounds

namespace VeriTile.Bench.Repairs.BgmvShrinkKernel.Correct
open VeriTile.Triton
noncomputable section

/-- Complete source events for the single 64-split grid and the independent
caller-coordinate K-term GEMV in every actual per-cell atomic order. The
ordinary runtime trace states only individual local atomic transitions. -/
theorem checked_real (x w y i : Contract.TensorInfo) (s : Contract.Scale) (indices : Nat → Int)
    (ok : Contract.Admitted x w y i s indices) (u : BlockState)
    (orders : Direct.Workset.Orders (Contract.call x w y i s indices ok))
    (run : Direct.CheckedReal.Run (Contract.call x w y i s indices ok) orders u indices) :
    Direct.CheckedReal.SourceWorkset (Contract.call x w y i s indices ok) u indices ∧
    ∀ b n,run.output b n=Direct.CheckedReal.math (Contract.call x w y i s indices ok) u indices b n :=
  Direct.CheckedReal.complete x w y i s indices ok u orders run

#axiomsClean checked_real
#axiomsClean Direct.Workset.fiber
#axiomsClean Direct.Bounds.formed
#axiomsClean Direct.Bounds.signed_offsets
#specNonCircular Direct.CheckedReal.math avoiding [Direct.CheckedReal.Run,Direct.RealExecution.contribution,Direct.Kernels.source,VeriTile.Triton.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BgmvShrinkKernel.Correct

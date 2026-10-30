import bench.tritonbench_g.rmsnorm_fused_llama.IO
import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.RmsnormFusedLlama.Correct
open VeriTile Triton
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- Complete canonical row grid of the selected FP32 source or its integer
index candidate. Channels cover the whole matrix and weight vector; the source
retains both loops and its output cast. Real erases precision mathematically. -/
abbrev kernelIO := IO.grid

/-- Independent RMS formula over every logical column. The Real domain retains
arbitrary row/column counts, including empty dimensions and multiple source
blocks. Only output/input region separation is required; X and W may alias. -/
specification rmsnorm_kernel_real (candidate : Bool) (X Y W : RegionName)
    (rows columns : Nat) (epsilon : ℝ) (xy : X≠Y) (wy : W≠Y) :
    Spec.Real (IO.logicalIO candidate X Y W rows columns 16384 epsilon ⊨
      fun values cell =>
        let xs := values.1
        let weights := values.2
        let row := cell.1
        let col := cell.2
        xs row col *
          (Real.sqrt ((∑ k : Fin columns, (xs row k)^2)/(columns:ℝ)+epsilon))⁻¹ *
          weights col) :=
  (RealPublicIO.logical_iff candidate X Y W rows columns 16384 epsilon).mpr
    (RealPublicIO.implements candidate X Y W rows columns 16384 epsilon (by decide) xy wy)

#axiomsClean rmsnorm_kernel_real
#specNonCircular RealGrid.value avoiding [kernel,RealGrid.run]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.RmsnormFusedLlama.Correct

import bench.tritonbench_g.rms_norm_triton.Kernels
import bench.tritonbench_g.rms_norm_triton.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.RmsNormTriton.Correct
open VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton FP.Structural Whole
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- All rows in any complete launch order are safe and terminate, every logical
output has the independent RMSNorm value, and all exterior typed cells remain
unchanged. X and W may alias each other; each remains separate from Y. -/
specification kernel_correct (c : Layout) (eps : ℝ)
    (xSeparate : c.X≠c.Y) (wSeparate : c.W≠c.Y)
    (rows : List Nat) (schedule : rows.Perm (List.range c.M)) :
    Spec.Real (LogicalIO.io c eps rows ⊨ fun values row col =>
      let xs := values.1
      let weights := values.2
      xs row col * (1 / Real.sqrt
        ((∑ k : Fin c.N, xs row k*xs row k)/(c.N:ℝ)+eps)) * weights col) :=
  (LogicalIO.logical_iff c eps rows).mpr
    (GridPublic.implements realAlgebra id (fun _ => id) c eps rows schedule
      xSeparate wSeparate (realFormula c.N eps) (rowValue_real c eps))

#specNonCircular formula avoiding [RmsNormTriton.kernel,rowValue,sourceRun,runRows]
#axiomsClean kernel_correct
#stmtSurfaceSubset kernel_correct ⊆
  [Layout,Layout.X,Layout.W,Layout.Y,Layout.M,Layout.N,RegionName,Spec.Real,
   LogicalIO.io,LogicalProgramIO.Implements]

end
end VeriTile.Bench.Repairs.RmsNormTriton.Correct

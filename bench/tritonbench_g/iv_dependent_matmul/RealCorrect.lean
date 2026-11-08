import bench.tritonbench_g.iv_dependent_matmul.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Correct
open VeriTile.Triton FP.Structural
open IvDependentMatmul
open scoped LogicalProgramIO
noncomputable section


/-- Complete selected source grid, including the empty inner sum. -/
specification kernel_real (rest : Algebra ℝ) (g : Kernel.Geometry)
    (layout : IO.Layout g) (launchable : Index64.Launchable g.request) :
    Spec.Real (IO.logicalIO (RealExecution.backend rest) .source g ⊨ fun xs row col =>
      ∑ k : Fin g.inner, xs.a row k * xs.b k col) :=
  (LogicalReader.full_judgment_iff rest g).mpr (LogicalReader.kernel_real rest g layout launchable)
#axiomsClean kernel_real

end
end VeriTile.Bench.Repairs.IvDependentMatmulDotShape.Correct

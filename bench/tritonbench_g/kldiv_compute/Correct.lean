import bench.tritonbench_g.kldiv_compute.Proofs.LogicalReader

namespace VeriTile.Bench.Repairs.KldivCompute.Kernel.Correct
open VeriTile.Triton FP.Structural
open scoped LogicalProgramIO
noncomputable section


/-- Every element of the complete logical grid, with source safety and exterior frame. -/
specification correctness (p : Parameters) (positive : 0<p.block) :
    Spec.Real (PublicIO.logicalIO Math.algebra p ⊨ fun xs i => xs.x i * Real.log (xs.x i / xs.y i)) :=
  (LogicalReader.full_judgment_iff p).mpr (PublicIO.implements Math.algebra p positive Math.formula Math.scalar_formula)
#axiomsClean correctness

end
end VeriTile.Bench.Repairs.KldivCompute.Kernel.Correct

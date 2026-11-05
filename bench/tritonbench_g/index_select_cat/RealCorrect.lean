import bench.tritonbench_g.index_select_cat.IO
import bench.tritonbench_g.index_select_cat.Proofs.LogicalCoordinates

namespace VeriTile.Bench.Repairs.IndexSelectCat.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- One BI×BC program copies the payload channel selected by the signed
row metadata. `gatherIO.read` maps that channel to
indices[i]*rowStride + column[j]*colStride; both logical tails are masked.
The result below is the initial selected source value at the same (i,j). -/
specification kernel_correct (rest : Algebra ℝ) (p : Layout) (rowPid colPid : Nat)
    (unique : LogicalIO.Unique p rowPid colPid) :
    Spec.Real (logicalGatherIO rest p rowPid colPid ⊨ fun xs i j => xs.payload i j) :=
  logical_gather_correct rest p rowPid colPid unique

#axiomsClean kernel_correct
#specNonCircular LogicalIO.read avoiding [kernel, FP.Structural.exec]
#specNonCircular copyValues avoiding [kernel, FP.Structural.exec]
#stmtSurfaceSubset kernel_correct ⊆
  [Spec.Real, LogicalProgramIO.Implements, logicalGatherIO, LogicalIO.Unique,
   Layout, FP.Structural.Algebra, GatherValues, GatherValues.payload, Layout.BI, Layout.BC]
end
end VeriTile.Bench.Repairs.IndexSelectCat.Correct

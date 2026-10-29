import bench.repairs.tritonbench_g.bmm_optimized_grid_axes_checked.Proofs.LogicalReader
import VeriTile.Meta.Specification

namespace VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked
open VeriTile Triton FP.Structural BmmOptimized Interface
open scoped BigOperators VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Complete source-grid batched matrix multiplication at batch/row/column
coordinates. Mathematical array extensions retain empty shapes and all tails;
the IO includes actual access safety and the complete exterior frame. -/
specification kernel_real (s : Shape) (c : Config) (rest : Algebra ℝ)
    (ps : List Program) (schedule : ps.Perm (programs s c)) :
    Spec.Real (logicalIO s c rest ps ⊨ fun xs batch row col =>
      ∑ k ∈ Finset.range s.k, xs.aAt batch.val row.val k * xs.bAt batch.val k col.val) :=
  logical_implements s c rest ps schedule

#axiomsClean kernel_real
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BmmOptimizedGridAxesChecked

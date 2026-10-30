import bench.repairs.tritonbench_g.mean_reduction.Proofs.RealIO

namespace VeriTile.Bench.Repairs.MeanReduction.Correctness
open VeriTile.Triton
open MeanIO (meanIO)
open scoped VeriTile.Triton.LogicalScheduledKernelIO
noncomputable section

/-- Every row of the complete source grid satisfies the independent finite
mean formula. The IO descriptor exposes the compressed input/output windows;
actual source access safety, success and the full exterior frame are proved.
Empty grids, N=0 and arbitrary complete launch order retain their prior scope. -/
specification mean_grid_real (M N : Nat) (tiles : List Nat)
    (schedule : tiles.Perm (List.range (gridSize M))) :
    Spec.Real (meanIO M N tiles ⊨ fun xs row =>
      (∑ col : Fin N, xs row col) / (N : ℝ)) := by
  exact MeanIO.logical_implements M N tiles schedule

#stmtSurfaceSubset mean_grid_real ⊆
  [Spec.Real,MeanIO.meanIO,LogicalScheduledKernelIO.Implements,gridSize]
#specNonCircular MeanIO.mathematical avoiding
  [kernel,exec,runAlgorithmSchedule]
#axiomsClean mean_grid_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MeanReduction.Correctness

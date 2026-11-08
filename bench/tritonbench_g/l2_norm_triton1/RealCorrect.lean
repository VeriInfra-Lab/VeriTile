import bench.tritonbench_g.l2_norm_triton1.Proofs.LogicalIO

/-! Reader entry: public full-grid IO and independent mathematical L2 target.
The source/runtime ABI, channels, masks and dense windows are in `IO.lean`.
Raw row and arbitrary-heap/byte corollaries are in `Misc/RealKernelResults.lean`.
-/
namespace VeriTile.Bench.Repairs.L2NormTriton1.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Every row of the actual grid is safe and succeeds, every logical output is
x/sqrt(sum(x²)+eps), and all exterior cells are preserved. -/
specification grid_correct (rows N B : Nat) (cover : N ≤ B) (eps : ℝ) :
    Spec.Real (LogicalIO.io RealExecution.algebra rows N B eps ⊨ fun x row col =>
      x row col / Real.sqrt ((∑ j : Fin N, (x row j)^2)+eps)) :=
  LogicalIO.implements rows N B cover eps

#axiomsClean grid_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.Correct

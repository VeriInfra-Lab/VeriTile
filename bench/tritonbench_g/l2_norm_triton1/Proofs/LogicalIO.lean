import bench.tritonbench_g.l2_norm_triton1.IO
import bench.tritonbench_g.l2_norm_triton1.Proofs.GridIO

namespace VeriTile.Bench.Repairs.L2NormTriton1.LogicalIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Exact equality with the previous full-grid judgment, preserving its schedule,
input binding, safety, output masks and exterior frame. -/
theorem judgment_iff (rows N B : Nat) (eps : ℝ) :
    (io RealExecution.algebra rows N B eps ⊨ fun x row col =>
      x row col / Real.sqrt ((∑ j : Fin N, (x row j)^2)+eps)) ↔
    (GridIO.gridIO RealExecution.algebra rows N B eps).Implements (GridIO.mathematical rows N eps) := by
  change (GridIO.gridIO RealExecution.algebra rows N B eps).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext xs o j
  fin_cases o
  dsimp [LogicalProgramIO.lift,io,inputs,outputs]
  simp [LogicalProgramIO.lift,io,inputs,outputs,GridIO.mathematical,GridIO.formula,GridIO.outputValues,
    ← Fin.sum_univ_eq_sum_range,GridIO.rowValues]

theorem implements (rows N B : Nat) (cover : N≤B) (eps : ℝ) :
    io RealExecution.algebra rows N B eps ⊨ fun x row col =>
      x row col / Real.sqrt ((∑ j : Fin N, (x row j)^2)+eps) :=
  (judgment_iff rows N B eps).mpr (GridIO.real_implements rows N B cover eps)
end
end VeriTile.Bench.Repairs.L2NormTriton1.LogicalIO

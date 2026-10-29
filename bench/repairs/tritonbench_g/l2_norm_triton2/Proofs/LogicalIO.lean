import bench.repairs.tritonbench_g.l2_norm_triton2.IO
import bench.repairs.tritonbench_g.l2_norm_triton2.Proofs.GridIO
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.LogicalIO

namespace VeriTile.Bench.Repairs.L2NormTriton2.LogicalIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem judgment_iff (rows N B : Nat) (eps : ℝ) :
    (io RealExecution.algebra rows N B eps ⊨ fun a row col =>
      let q := (∑ j : Fin N, (a.x row j)^2)+eps
      let dot := ∑ j : Fin N, a.dy row j*a.x row j
      a.dy row col / Real.sqrt q-dot/q/Real.sqrt q*a.x row col) ↔
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
    io RealExecution.algebra rows N B eps ⊨ fun a row col =>
      let q := (∑ j : Fin N, (a.x row j)^2)+eps
      let dot := ∑ j : Fin N, a.dy row j*a.x row j
      a.dy row col / Real.sqrt q-dot/q/Real.sqrt q*a.x row col :=
  (judgment_iff rows N B eps).mpr (GridIO.real_implements rows N B cover eps)
end
end VeriTile.Bench.Repairs.L2NormTriton2.LogicalIO

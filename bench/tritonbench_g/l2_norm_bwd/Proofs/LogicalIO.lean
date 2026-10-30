import bench.tritonbench_g.l2_norm_bwd.IO
import bench.tritonbench_g.l2_norm_bwd.Proofs.IOGrid
import bench.tritonbench_g.l2_norm_bwd.Proofs.Real

namespace VeriTile.Bench.Repairs.L2NormBwd.LogicalIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem judgment_iff (layout : Strides) (rows N B : Nat) (eps : ℝ) :
    (io Math.algebra layout rows N B eps ⊨ fun a row col =>
      let q := (∑ j : Fin N, a.x row j*a.x row j)+eps
      let dot := ∑ j : Fin N, a.dy row j*a.x row j
      a.dy row col*(1/Real.sqrt q)-dot*(1/q)*(1/Real.sqrt q)*a.x row col) ↔
    (PublicIO.sourceIO Math.algebra id (fun _ => id) layout rows N B eps).Implements
      (fun xs => PublicIO.result rows N (fun row col => Math.derivative N eps
        (PublicIO.rowValues xs Bool.false row) (PublicIO.rowValues xs Bool.true row) col.val)) := by
  change (PublicIO.sourceIO Math.algebra id (fun _ => id) layout rows N B eps).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext xs o j
  fin_cases o
  dsimp [LogicalProgramIO.lift,io,inputs,outputs]
  simp [LogicalProgramIO.lift,io,inputs,outputs,PublicIO.result,Math.derivative,PublicIO.rowValues]
  rfl

theorem implements (layout : Strides) (rows N B : Nat)
    (cover : N≤B) (unique : PublicIO.OutputUnique layout rows N) (eps : ℝ) :
    io Math.algebra layout rows N B eps ⊨ fun a row col =>
      let q := (∑ j : Fin N, a.x row j*a.x row j)+eps
      let dot := ∑ j : Fin N, a.dy row j*a.x row j
      a.dy row col*(1/Real.sqrt q)-dot*(1/q)*(1/Real.sqrt q)*a.x row col := by
  apply (judgment_iff layout rows N B eps).mpr
  exact PublicIO.implements Math.algebra id (fun _ => id) layout rows N B cover unique
    eps (Math.derivative N eps) (fun xs ys col _ => Math.source_derivative N B cover eps xs ys col)
end
end VeriTile.Bench.Repairs.L2NormBwd.LogicalIO

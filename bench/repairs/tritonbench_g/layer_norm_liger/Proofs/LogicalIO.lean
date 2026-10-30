import bench.repairs.tritonbench_g.layer_norm_liger.IO
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.RealInterface

namespace VeriTile.Bench.Repairs.LayerNormLiger.LogicalIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

theorem forward_iff (rest : Algebra ℝ) (p : Forward.Parameters) (pid : Nat) :
    (LogicalForwardIO.io rest p pid ⊨ fun a => ⟨
      fun i => Logical.y p.columns p.epsilon (LogicalForwardIO.extend a.x)
        (LogicalForwardIO.extend a.weight) (LogicalForwardIO.extend a.bias) i.val,
      Logical.mean p.columns (LogicalForwardIO.extend a.x),
      Logical.rstd p.columns p.epsilon (LogicalForwardIO.extend a.x)⟩) ↔
    (LayerIO.forwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Implements
      (LayerIO.forwardFormula p) := by
  change (LayerIO.forwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext vals o j
  fin_cases o <;> dsimp [LogicalProgramIO.lift,LogicalForwardIO.io,LogicalForwardIO.inputs,LogicalForwardIO.outputs]
  all_goals simp [LayerIO.forwardFormula,LayerIO.fAt,LayerIO.fInput,LogicalForwardIO.extend]
  all_goals rfl

theorem backward_iff (rest : Algebra ℝ) (p : Backward.Parameters) (pid : Nat) :
    (LogicalBackwardIO.io rest p pid ⊨ fun a => ⟨
      fun row col => Logical.dx p.columns (LogicalBackwardIO.extend2 a.x row.val)
        (LogicalBackwardIO.extend a.weight) (LogicalBackwardIO.extend2 a.dy row.val)
        (LogicalBackwardIO.extend a.mean row.val) (LogicalBackwardIO.extend a.inverse row.val) col.val,
      fun col => ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows),
        LogicalBackwardIO.extend2 a.dy row col.val *
          ((LogicalBackwardIO.extend2 a.x row col.val-LogicalBackwardIO.extend a.mean row)*LogicalBackwardIO.extend a.inverse row),
      fun col => ∑ row ∈ Finset.Ico (pid*p.rowsPerProgram) (min ((pid+1)*p.rowsPerProgram) p.rows),
        LogicalBackwardIO.extend2 a.dy row col.val⟩) ↔
    (LayerIO.backwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Implements
      (LayerIO.backwardFormula p pid) := by
  change (LayerIO.backwardIO (Logical.algebra rest) Logical.sumPrimitive p pid).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext vals o j
  fin_cases o <;> dsimp [LogicalProgramIO.lift,LogicalBackwardIO.io,LogicalBackwardIO.inputs,LogicalBackwardIO.outputs]
  all_goals simp [LayerIO.backwardFormula,LayerIO.dwFormula,LayerIO.dbFormula,LayerIO.first,LayerIO.last,
    LayerIO.bx,LayerIO.bw,LayerIO.bdy,LayerIO.bmean,LayerIO.brstd,Logical.xhat,
    LogicalBackwardIO.extend,LogicalBackwardIO.extend2]
  all_goals try rfl
  all_goals
    change @Eq ℝ _ _
    apply Finset.sum_congr rfl
    intro row member
    have hr : row<p.rows := lt_of_lt_of_le (Finset.mem_Ico.mp member).2 (min_le_right _ _)
    simp [hr]
end
end VeriTile.Bench.Repairs.LayerNormLiger.LogicalIO

import bench.repairs.tritonbench_g.layer_norm_ops.IO
import bench.repairs.tritonbench_g.layer_norm_ops.Proofs.IOReal

namespace VeriTile.Bench.Repairs.LayerNormOps.LogicalIO
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000

@[simp] theorem extend2_row {m n : Nat} (x : Fin m → Fin n → ℝ) (row : Fin m) :
    LogicalBackwardIO.extend2 x row.val = Interface.extend (x row) := by
  funext col
  simp [LogicalBackwardIO.extend2,Interface.extend]
  rfl

def forwardFormula (p : ForwardParameters) (a : LogicalForwardIO.Inputs p) : LogicalForwardIO.Outputs p :=
  let x := fun j => LogicalForwardIO.extend a.x j + if p.hasResidual then LogicalForwardIO.extend a.residual j else 0
  ⟨fun i => Logical.y p.columns p.rms p.hasBias p.epsilon x
      (LogicalForwardIO.extend a.weight) (LogicalForwardIO.extend a.bias) i.val,
   fun i => x i.val,
   Logical.mean p.columns p.rms x,
   Logical.rstd p.columns p.rms p.epsilon x⟩

def backwardFormula (p : BackwardParameters) (pid : Nat) (a : LogicalBackwardIO.Inputs p) : LogicalBackwardIO.Outputs p :=
  let dx := fun row col => Logical.dx p.columns p.rms p.hasDres
    (LogicalBackwardIO.extend2 a.x row.val) (LogicalBackwardIO.extend a.weight)
    (LogicalBackwardIO.extend2 a.dy row.val) (LogicalBackwardIO.extend2 a.dres row.val)
    (LogicalBackwardIO.extend a.mean row.val) (LogicalBackwardIO.extend a.inverse row.val) col.val
  ⟨dx,
   fun row col => ((LogicalBackwardIO.extend2 a.x row.val col.val -
      if p.rms then 0 else LogicalBackwardIO.extend a.mean row.val)*LogicalBackwardIO.extend a.inverse row.val)*LogicalBackwardIO.extend a.weight col.val+
      if p.hasBias then LogicalBackwardIO.extend a.bias col.val else 0,
   dx,
   fun col => ∑ row ∈ Finset.Ico (Interface.rowStart p pid) (Interface.rowStop p pid),
      LogicalBackwardIO.extend2 a.dy row col.val * ((LogicalBackwardIO.extend2 a.x row col.val-
        if p.rms then 0 else LogicalBackwardIO.extend a.mean row)*LogicalBackwardIO.extend a.inverse row),
   fun col => ∑ row ∈ Finset.Ico (Interface.rowStart p pid) (Interface.rowStop p pid),
      LogicalBackwardIO.extend2 a.dy row col.val⟩


theorem forward_iff (p : ForwardParameters) (pid : Nat) :
    (LogicalForwardIO.io Interface.realAlgebra p pid ⊨ forwardFormula p) ↔
    (Interface.forwardIO Interface.realAlgebra p pid).Implements (Interface.forwardMath p) := by
  change (Interface.forwardIO Interface.realAlgebra p pid).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext vals o j
  fin_cases o <;> dsimp [LogicalProgramIO.lift,LogicalForwardIO.io,LogicalForwardIO.inputs,LogicalForwardIO.outputs]
  all_goals simp [forwardFormula,Interface.forwardMath,Interface.forwardMathInput,Interface.forwardResult,
    Interface.vector,Interface.forwardX,Interface.forwardWeight,Interface.forwardBias,Interface.forwardResidual,
    Interface.extend,LogicalForwardIO.extend]
  all_goals rfl

theorem backward_iff (p : BackwardParameters) (pid : Nat) :
    (LogicalBackwardIO.io Interface.realAlgebra p pid ⊨ backwardFormula p pid) ↔
    (Interface.backwardIO Interface.realAlgebra p pid).Implements (Interface.backwardMath p pid) := by
  change (Interface.backwardIO Interface.realAlgebra p pid).Implements _ ↔ _
  apply Iff.of_eq
  congr 1
  funext vals o j
  fin_cases o <;> dsimp [LogicalProgramIO.lift,LogicalBackwardIO.io,LogicalBackwardIO.inputs,LogicalBackwardIO.outputs]
  all_goals simp [backwardFormula,Interface.backwardMath,Interface.backwardResult,Interface.backwardMathDX,
    Interface.backwardMathY,Interface.backwardMathDW,Interface.backwardMathDB,Interface.backwardMathHat,
    Interface.vector,Interface.backwardX,Interface.backwardWeight,Interface.backwardBias,Interface.backwardDY,
    Interface.backwardDres,Interface.backwardMean,Interface.backwardInverse,
    Interface.extend,LogicalBackwardIO.extend,LogicalBackwardIO.extend2]
  all_goals try rfl
  all_goals
    change @Eq ℝ _ _
    apply Finset.sum_congr rfl
    intro row member
    have hr : row<p.rows := lt_of_lt_of_le (Finset.mem_Ico.mp member).2 (by
      exact min_le_right _ _)
    simp [hr]
    rfl
end
end VeriTile.Bench.Repairs.LayerNormOps.LogicalIO

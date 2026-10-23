import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.LogicalReader
namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped BigOperators
open scoped LogicalProgramIO
noncomputable section

def forwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (log causal : Bool) (mode : Forward.MaskMode)
    (xs : Interface.LogicalInputs (WithBot ℝ) B Q D) (batch : Fin B) (row : Fin Q) (col : Fin D) : WithBot ℝ :=
  let scores := Interface.score K causal mode xs batch row
  let partition := ∑ j : Fin K, if (Interface.logicalScore K covered causal mode xs batch row j).isSome then
    Real.exp ((Interface.logicalScore K covered causal mode xs batch row j).getD 0) else 0
  if (scores col).isSome then some (if log then (scores col).getD 0-Real.log partition
    else Real.exp ((scores col).getD 0)/partition) else if log then none else some 0

def backwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (log causal : Bool)
    (xs : Interface.LogicalInputs (WithBot ℝ) B Q D) (batch : Fin B) (row : Fin Q) (col : Fin D) : WithBot ℝ := some (
  let g := fun j : Fin D => if Backward.readActive K row causal j.val then (xs.left batch row j).getD 0 else 0
  let o := fun j : Fin D => if Backward.readActive K row causal j.val then (xs.right batch row j).getD 0 else 0
  let logicalG := fun j : Fin K => if Backward.readActive K row causal j.val then
    (Interface.logicalLeft K covered xs batch row j).getD 0 else 0
  let logicalO := fun j : Fin K => if Backward.readActive K row causal j.val then
    (Interface.logicalRight K covered xs batch row j).getD 0 else 0
  if causal && decide (row.val < col.val) then 0 else if log then
    g col-Real.exp (o col)*(∑ j : Fin K, logicalG j)
  else o col*(g col-∑ j : Fin K, logicalG j*logicalO j))

def logBackwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (causal : Bool)
    (xs : Interface.LogicalInputs (WithBot ℝ) B Q D) (batch : Fin B) (row : Fin Q) (col : Fin D) : WithBot ℝ := some (
  if causal && decide (row.val < col.val) then 0 else
    (if Forward.readActive K row causal col.val then (xs.left batch row col).getD 0 else 0) -
    (if Forward.readActive K row causal col.val then
      match xs.right batch row col with | none => 0 | some x => Real.exp x else 1) *
    ∑ j : Fin K, if Forward.readActive K row causal j.val then
      (Interface.logicalLeft K covered xs batch row j).getD 0 else 0)

/-- Complete selected row grid, logical K reductions, and the original mathematical domain. -/
specification forward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (covered : K≤D) (log causal : Bool) (mode : Forward.MaskMode) :
    Spec.Real (Interface.logicalIO Forward.TypedExecution.model (.forward log causal mode) B Q K D ⊨{Interface.logicalForwardDomain K causal mode} forwardFormula K covered log causal mode) :=
  (LogicalReader.forward_full_iff B Q K D covered log causal mode).mpr
    (LogicalReader.forward_kernel_real_raw B Q K D hD covered log causal mode)
#axiomsClean forward_kernel_real_raw

/-- Complete selected row grid, logical K reductions, and the original mathematical domain. -/
specification backward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (covered : K≤D) (log causal : Bool) :
    Spec.Real (Interface.logicalIO Forward.TypedExecution.model (.backward log causal) B Q K D ⊨{Interface.logicalBackwardDomain K causal Bool.false} backwardFormula K covered log causal) :=
  (LogicalReader.backward_full_iff B Q K D covered log causal).mpr
    (LogicalReader.backward_kernel_real_raw B Q K D hD covered log causal)
#axiomsClean backward_kernel_real_raw

/-- Complete selected row grid, logical K reductions, and the original mathematical domain. -/
specification log_backward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (covered : K≤D) (causal : Bool) :
    Spec.Real (Interface.logicalIO Forward.TypedExecution.model (.backward Bool.true causal) B Q K D ⊨{Interface.logicalBackwardDomain K causal Bool.true} logBackwardFormula K covered causal) :=
  (LogicalReader.log_backward_full_iff B Q K D covered causal).mpr
    (LogicalReader.log_backward_kernel_real_raw B Q K D hD covered causal)
#axiomsClean log_backward_kernel_real_raw

end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.Correct

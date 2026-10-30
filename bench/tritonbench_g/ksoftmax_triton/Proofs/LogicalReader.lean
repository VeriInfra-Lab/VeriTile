import VeriTile.Triton.Float.LogicalProgramIO
import bench.tritonbench_g.ksoftmax_triton.IO
import bench.tritonbench_g.ksoftmax_triton.Proofs.LogicalWidth

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped ProgramIO BigOperators
noncomputable section

/-- The complete selected row grid and exact causal/mask read windows are
shown in IO.lean. The source fixes FP32 precision; Real is mathematical. -/
def realIO (op : Operation) (B Q K D : Nat) : ProgramIO (WithBot ℝ) (WithBot ℝ) :=
  Interface.programIO Forward.TypedExecution.model op B Q K D (Interface.selectedRows B Q K)

/-- Extended scores retain causal and optional-mask semantics. The partition
sums exactly the logical K columns, embedded in the padded D channel by covered. -/
def forwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (log causal : Bool)
    (mode : Forward.MaskMode) (v : Interface.Inputs (WithBot ℝ) B Q D) :
    Interface.Outputs (WithBot ℝ) B Q D :=
  Interface.result fun b r i =>
    let scores := Interface.forwardScores K causal mode v b r
    let partition := ∑ j : Fin K, if (scores (Fin.castLE covered j)).isSome then
      Real.exp ((scores (Fin.castLE covered j)).getD 0) else 0
    if (scores i).isSome then some (if log then (scores i).getD 0-Real.log partition
      else Real.exp ((scores i).getD 0)/partition)
    else if log then none else some 0

/-- The ordinary-Real Jacobian-vector product, with logical K reductions.
Saved values are probabilities for softmax and log probabilities for log-softmax. -/
def backwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (log causal : Bool)
    (v : Interface.Inputs (WithBot ℝ) B Q D) : Interface.Outputs (WithBot ℝ) B Q D :=
  Interface.result fun b r i => some (
    let g := fun j => if Backward.readActive K r causal j.val then
      (Interface.left v b r j).getD 0 else 0
    let o := fun j => if Backward.readActive K r causal j.val then
      (Interface.right v b r j).getD 0 else 0
    if causal && decide (r.val < i.val) then 0 else if log then
      g i-Real.exp (o i)*(∑ j : Fin K, g (Fin.castLE covered j))
    else o i*(g i-∑ j : Fin K, g (Fin.castLE covered j)*o (Fin.castLE covered j)))

/-- The broader log-backward treats saved negative infinity as probability
zero, keeps causal future outputs zero, and also sums only the logical K lanes. -/
def logBackwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (causal : Bool)
    (v : Interface.Inputs (WithBot ℝ) B Q D) : Interface.Outputs (WithBot ℝ) B Q D :=
  Interface.result fun b r i => some (
    if causal && decide (r.val < i.val) then 0 else
      (if Forward.readActive K r causal i.val then (Interface.left v b r i).getD 0 else 0) -
      (if Forward.readActive K r causal i.val then
        match Interface.right v b r i with | none => 0 | some x => Real.exp x
       else 1) *
      ∑ j : Fin K, if Forward.readActive K r causal j.val then
        (Interface.left v b r (Fin.castLE covered j)).getD 0 else 0)




/-- Independent normalized exponential/logarithmic formula on extended Real
scores, with at least one finite score per nonempty row. -/
theorem forward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (covered : K≤D)
    (log causal : Bool) (mode : Forward.MaskMode) :
    Spec.Real (realIO (.forward log causal mode) B Q K D ⊨{Interface.ForwardDomain K causal mode}
      forwardFormula K covered log causal mode) :=
  by
    have target : @forwardFormula B Q D K covered log causal mode = Interface.forwardFormula K log causal mode := by
      funext v p
      rcases p with ⟨port,hp⟩
      have hz : port=0 := by have small : port<1 := hp; omega
      subst port
      funext j
      have partition := LogicalWidth.sum_restrict covered
        (fun i => if (Interface.forwardScores K causal mode v j.1 j.2.1 i).isSome then
          Real.exp ((Interface.forwardScores K causal mode v j.1 j.2.1 i).getD 0) else 0)
        (fun i hi => by simp only [LogicalWidth.scores_tail causal mode v j.1 j.2.1 i hi]; rfl)
      simp only [forwardFormula,Interface.forwardFormula,Interface.result,KernelExtended.formula,softmax,
        VeriTile.Bench.Repairs.Ksoftmax.partition,partition]
    simpa only [target] using Interface.forward_implements B Q K D hD covered log causal mode

/-- Ordinary-Real backward scope, covering both softmax and log-softmax.
These mathematical domain conditions do not restrict native FP32 words. -/
theorem backward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (covered : K≤D)
    (log causal : Bool) :
    Spec.Real (realIO (.backward log causal) B Q K D ⊨{Interface.BackwardDomain K causal Bool.false}
      backwardFormula K covered log causal) :=
  by
    have target : @backwardFormula B Q D K covered log causal = Interface.backwardFormula K log causal := by
      funext v p
      rcases p with ⟨port,hp⟩
      have hz : port=0 := by have small : port<1 := hp; omega
      subst port
      funext j
      have sg := LogicalWidth.sum_restrict covered
        (fun i => if Backward.readActive K j.2.1 causal i.val then
          (Interface.left v j.1 j.2.1 i).getD 0 else 0)
        (fun i hi => by simp only [LogicalWidth.backwardActive_tail K j.2.1 causal i.val hi]; rfl)
      have sgo := LogicalWidth.sum_restrict covered
        (fun i => (if Backward.readActive K j.2.1 causal i.val then
          (Interface.left v j.1 j.2.1 i).getD 0 else 0) *
          (if Backward.readActive K j.2.1 causal i.val then (Interface.right v j.1 j.2.1 i).getD 0 else 0))
        (fun i hi => by simp only [LogicalWidth.backwardActive_tail K j.2.1 causal i.val hi]; simp)
      simp only [backwardFormula,Interface.backwardFormula,Interface.result,gradient,sg,sgo]
    simpa only [target] using Interface.backward_implements B Q K D hD covered log causal

/-- The broader log-backward formula permits saved values of negative
infinity and uses exp(-infinity)=0. Causal future coordinates are zero. -/
theorem log_backward_kernel_real_raw (B Q K D : Nat) (hD : 0<D) (covered : K≤D)
    (causal : Bool) :
    Spec.Real (realIO (.backward Bool.true causal) B Q K D ⊨{Interface.BackwardDomain K causal Bool.true}
      logBackwardFormula K covered causal) :=
  by
    have target : @logBackwardFormula B Q D K covered causal = Interface.logBackwardFormula K causal := by
      funext v p
      rcases p with ⟨port,hp⟩
      have hz : port=0 := by have small : port<1 := hp; omega
      subst port
      funext j
      have sg := LogicalWidth.sum_restrict covered
        (fun i => if Forward.readActive K j.2.1 causal i.val then
          (Interface.left v j.1 j.2.1 i).getD 0 else 0)
        (fun i hi => by simp only [LogicalWidth.readActive_tail K j.2.1 causal i.val hi]; rfl)
      simp only [logBackwardFormula,Interface.logBackwardFormula,Interface.result,
        KernelLogBackward.formula,KernelLogBackward.probability,sg]
      rfl
    simpa only [target] using Interface.log_backward_implements B Q K D hD covered causal

#stmtSurfaceSubset log_backward_kernel_real_raw ⊆
  [Spec.Real, realIO, ProgramIO.ImplementsOn, Interface.BackwardDomain, logBackwardFormula, Operation.backward]
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.LogicalReader

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.LogicalReader
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped BigOperators
open scoped LogicalProgramIO
noncomputable section

def logicalForwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (log causal : Bool) (mode : Forward.MaskMode)
    (xs : Interface.LogicalInputs (WithBot ℝ) B Q D) (batch : Fin B) (row : Fin Q) (col : Fin D) : WithBot ℝ :=
  let scores := Interface.score K causal mode xs batch row
  let partition := ∑ j : Fin K, if (Interface.logicalScore K covered causal mode xs batch row j).isSome then
    Real.exp ((Interface.logicalScore K covered causal mode xs batch row j).getD 0) else 0
  if (scores col).isSome then some (if log then (scores col).getD 0-Real.log partition
    else Real.exp ((scores col).getD 0)/partition) else if log then none else some 0

def logicalBackwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (log causal : Bool)
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

def logicalLogBackwardFormula {B Q D : Nat} (K : Nat) (covered : K≤D) (causal : Bool)
    (xs : Interface.LogicalInputs (WithBot ℝ) B Q D) (batch : Fin B) (row : Fin Q) (col : Fin D) : WithBot ℝ := some (
  if causal && decide (row.val < col.val) then 0 else
    (if Forward.readActive K row causal col.val then (xs.left batch row col).getD 0 else 0) -
    (if Forward.readActive K row causal col.val then
      match xs.right batch row col with | none => 0 | some x => Real.exp x else 1) *
    ∑ j : Fin K, if Forward.readActive K row causal j.val then
      (Interface.logicalLeft K covered xs batch row j).getD 0 else 0)

theorem forward_full_iff (B Q K D : Nat) (covered : K≤D) (log causal : Bool) (mode : Forward.MaskMode) :
    (Interface.logicalIO Forward.TypedExecution.model (.forward log causal mode) B Q K D ⊨{Interface.logicalForwardDomain K causal mode} logicalForwardFormula K covered log causal mode) ↔
      (realIO (.forward log causal mode) B Q K D).ImplementsOn (Interface.ForwardDomain K causal mode) (forwardFormula K covered log causal mode) := by
  have domain : (fun raw => Interface.logicalForwardDomain K causal mode ((Interface.logicalIO Forward.TypedExecution.model (.forward log causal mode) B Q K D).inputs raw)) = Interface.ForwardDomain K causal mode := by
    funext raw
    unfold Interface.logicalIO Interface.logicalForwardDomain
    exact congrArg _ ((Interface.logicalInputs (WithBot ℝ) B Q D).left_inv raw)
  have target : (Interface.logicalIO Forward.TypedExecution.model (.forward log causal mode) B Q K D).lift (logicalForwardFormula K covered log causal mode) = forwardFormula K covered log causal mode := by
    funext raw o; fin_cases o; funext j; rfl
  change (realIO (.forward log causal mode) B Q K D).ImplementsOn
    (fun raw => Interface.logicalForwardDomain K causal mode ((Interface.logicalIO Forward.TypedExecution.model (.forward log causal mode) B Q K D).inputs raw))
    ((Interface.logicalIO Forward.TypedExecution.model (.forward log causal mode) B Q K D).lift (logicalForwardFormula K covered log causal mode)) ↔ _
  exact Iff.of_eq (congrArg₂ (fun d f => (realIO (.forward log causal mode) B Q K D).ImplementsOn d f) domain target)

theorem backward_full_iff (B Q K D : Nat) (covered : K≤D) (log causal : Bool) :
    (Interface.logicalIO Forward.TypedExecution.model (.backward log causal) B Q K D ⊨{Interface.logicalBackwardDomain K causal Bool.false} logicalBackwardFormula K covered log causal) ↔
      (realIO (.backward log causal) B Q K D).ImplementsOn (Interface.BackwardDomain K causal Bool.false) (backwardFormula K covered log causal) := by
  have domain : (fun raw => Interface.logicalBackwardDomain K causal Bool.false ((Interface.logicalIO Forward.TypedExecution.model (.backward log causal) B Q K D).inputs raw)) = Interface.BackwardDomain K causal Bool.false := by
    funext raw
    unfold Interface.logicalIO Interface.logicalBackwardDomain
    exact congrArg _ ((Interface.logicalInputs (WithBot ℝ) B Q D).left_inv raw)
  have target : (Interface.logicalIO Forward.TypedExecution.model (.backward log causal) B Q K D).lift (logicalBackwardFormula K covered log causal) = backwardFormula K covered log causal := by
    funext raw o; fin_cases o; funext j; rfl
  change (realIO (.backward log causal) B Q K D).ImplementsOn
    (fun raw => Interface.logicalBackwardDomain K causal Bool.false ((Interface.logicalIO Forward.TypedExecution.model (.backward log causal) B Q K D).inputs raw))
    ((Interface.logicalIO Forward.TypedExecution.model (.backward log causal) B Q K D).lift (logicalBackwardFormula K covered log causal)) ↔ _
  exact Iff.of_eq (congrArg₂ (fun d f => (realIO (.backward log causal) B Q K D).ImplementsOn d f) domain target)

theorem log_backward_full_iff (B Q K D : Nat) (covered : K≤D) (causal : Bool) :
    (Interface.logicalIO Forward.TypedExecution.model (.backward Bool.true causal) B Q K D ⊨{Interface.logicalBackwardDomain K causal Bool.true} logicalLogBackwardFormula K covered causal) ↔
      (realIO (.backward Bool.true causal) B Q K D).ImplementsOn (Interface.BackwardDomain K causal Bool.true) (logBackwardFormula K covered causal) := by
  have domain : (fun raw => Interface.logicalBackwardDomain K causal Bool.true ((Interface.logicalIO Forward.TypedExecution.model (.backward Bool.true causal) B Q K D).inputs raw)) = Interface.BackwardDomain K causal Bool.true := by
    funext raw
    unfold Interface.logicalIO Interface.logicalBackwardDomain
    exact congrArg _ ((Interface.logicalInputs (WithBot ℝ) B Q D).left_inv raw)
  have target : (Interface.logicalIO Forward.TypedExecution.model (.backward Bool.true causal) B Q K D).lift (logicalLogBackwardFormula K covered causal) = logBackwardFormula K covered causal := by
    funext raw o; fin_cases o; funext j; rfl
  change (realIO (.backward Bool.true causal) B Q K D).ImplementsOn
    (fun raw => Interface.logicalBackwardDomain K causal Bool.true ((Interface.logicalIO Forward.TypedExecution.model (.backward Bool.true causal) B Q K D).inputs raw))
    ((Interface.logicalIO Forward.TypedExecution.model (.backward Bool.true causal) B Q K D).lift (logicalLogBackwardFormula K covered causal)) ↔ _
  exact Iff.of_eq (congrArg₂ (fun d f => (realIO (.backward Bool.true causal) B Q K D).ImplementsOn d f) domain target)

end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.LogicalReader

import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.RowValue
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Graph
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedModel

/-! Removing historical extra store casts changes the generic-algebra
reference to match the actual source. The previously independent Native graph
and exact Real results remain equal for all inputs, flags and reduction plans.
No cast identity for arbitrary algebras is assumed. -/
namespace VeriTile.Bench.Repairs.Ksoftmax.PrecisionProfile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def previousForwardTail {α : Type} (M : Algebra α) (log : Bool) {D : Nat}
    (score : Fin D → α) (i : Fin D) : Cell α :=
  let z := fun j => M.binary (some .fp32) .real .sub (score j) (rowMax M score)
  let num := fun j => M.unary (some .fp32) .exp (z j)
  let den := rowSum M num
  Cell.mk .real (M.cast (some .fp32) .real .real
    (if log then M.binary (some .fp32) .real .sub (z i) (M.unary (some .fp32) .log den)
     else M.binary (some .fp32) .real .div (num i) den))

def previousForwardValue {α : Type} (M : Algebra α) (K row : Nat) (log causal : Bool)
    (mode : Forward.MaskMode) {D : Nat} (xs masks : Fin D → α) (i : Fin D) : Cell α :=
  let x := selectedValue M M.negInf K row causal xs
  let score := fun j => if mode.present then M.binary (some .fp32) .real .add (x j)
    (maskedValue M M.negInf K row causal masks j) else x j
  previousForwardTail M log score i

def previousBackwardValue {α : Type} (M : Algebra α) (K row : Nat) (log causal : Bool)
    {D : Nat} (gs os : Fin D → α) (i : Fin D) : Cell α :=
  let zero := M.literal (some .fp32) .real 0
  let g := selectedValue M zero K row causal gs
  let o := selectedValue M zero K row causal os
  let v := if log then M.binary (some .fp32) .real .sub (g i)
      (M.binary (some .fp32) .real .mul (M.unary (some .fp32) .exp (o i)) (rowSum M g))
    else M.binary (some .fp32) .real .mul (o i)
      (M.binary (some .fp32) .real .sub (g i)
        (rowSum M (fun j => M.binary (some .fp32) .real .mul (g j) (o j))))
  Cell.mk .real (M.cast (some .fp32) .real .real
    (if causal then if i.val≤row then v else zero else v))


theorem native_forward_target_unchanged (rest : Algebra Graph.Expr) (profile : Graph.Profile)
    {D : Nat} (K row : Nat) (log causal : Bool) (mode : Forward.MaskMode)
    (xs masks : Fin D → Graph.Expr) (i : Fin D) :
    previousForwardValue (Graph.algebra rest profile) K row log causal mode xs masks i =
      forwardValue (Graph.algebra rest profile) K row log causal mode xs masks i := rfl

theorem native_backward_target_unchanged (rest : Algebra Graph.Expr) (profile : Graph.Profile)
    {D : Nat} (K row : Nat) (log causal : Bool) (gs os : Fin D → Graph.Expr) (i : Fin D) :
    previousBackwardValue (Graph.algebra rest profile) K row log causal gs os i =
      backwardValue (Graph.algebra rest profile) K row log causal gs os i := rfl

theorem real_forward_target_unchanged {D : Nat} (K row : Nat)
    (log causal : Bool) (mode : Forward.MaskMode) (xs masks : Fin D → WithBot ℝ) (i : Fin D) :
    previousForwardValue Forward.TypedExecution.model K row log causal mode xs masks i =
      forwardValue Forward.TypedExecution.model K row log causal mode xs masks i := rfl

theorem real_backward_target_unchanged {D : Nat} (K row : Nat)
    (log causal : Bool) (gs os : Fin D → WithBot ℝ) (i : Fin D) :
    previousBackwardValue Forward.TypedExecution.model K row log causal gs os i =
      backwardValue Forward.TypedExecution.model K row log causal gs os i := rfl

#axiomsClean native_forward_target_unchanged
#axiomsClean native_backward_target_unchanged
#axiomsClean real_forward_target_unchanged
#axiomsClean real_backward_target_unchanged
end
end VeriTile.Bench.Repairs.Ksoftmax.PrecisionProfile

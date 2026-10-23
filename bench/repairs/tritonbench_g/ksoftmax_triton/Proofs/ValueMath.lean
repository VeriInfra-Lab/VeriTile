import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.RowValue
import bench.repairs.tritonbench_g.ksoftmax_triton.Proofs.Math
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedModel
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTileBridge

namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

theorem selected_real {D : Nat} (K row : Nat) (causal : Bool) (fallback : WithBot ℝ)
    (values : Fin D → WithBot ℝ) (i : Fin D) :
    selectedValue Forward.TypedExecution.model fallback K row causal values i=
      if Forward.readActive K row causal i.val then values i else fallback := by
  cases causal
  · simp [selectedValue,maskedValue,Forward.TypedExecution.model,Forward.readActive]
  · by_cases h : i.val≤row
    · simp [selectedValue,maskedValue,Forward.TypedExecution.model,Forward.readActive,h,Nat.not_lt.mpr h]
    · simp [selectedValue,maskedValue,Forward.TypedExecution.model,Forward.readActive,h,Nat.lt_of_not_ge h]

theorem rowMax_real {D : Nat} (hD : 0<D) (values : Fin D → WithBot ℝ) :
    rowMax Forward.TypedExecution.model values=Forward.rowMaximum values := by
  have h:=Forward.reduceMax_valuesTile hD values
  unfold rowMax Forward.TypedExecution.model
  change ((Tile.reduceMax (shape := [D]) ⟨0,by simp⟩ Bool.false (Forward.valuesTile values)).getD
    ⟨fun _ => none⟩).data PUnit.unit=Forward.rowMaximum values
  rw [h]
  rfl

theorem forwardTail_real {D : Nat} (hD : 0<D) (log : Bool)
    (values : Fin D → WithBot ℝ) (i : Fin D) :
    forwardTail Forward.TypedExecution.model log values i=
      Cell.mk .real ((Forward.resultTile log values).data (i,PUnit.unit)) := by
  unfold forwardTail
  rw [rowMax_real hD]
  cases log <;> rfl

/-- The reference becomes the independent unshifted softmax under exact
extended-real primitives; causal log outputs remain bottom, not real zero. -/
theorem forwardValue_real {D : Nat} (K row : Nat) (hK : 0<K) (hD : 0<D)
    (log causal : Bool) (mode : Forward.MaskMode) (input mask : Fin D → WithBot ℝ)
    (xs masks : Fin D → ℝ)
    (hx : ∀ j, Forward.readActive K row causal j.val=Bool.true → input j=some (xs j))
    (hm : ∀ j, mode.present=Bool.true → Forward.readActive K row causal j.val=Bool.true → mask j=some (masks j))
    (i : Fin D) :
    forwardValue Forward.TypedExecution.model K row log causal mode input mask i=
      Cell.mk .real (softmax (fun j => Forward.readActive K row causal j.val)
        (fun j => xs j+if mode.present then masks j else 0) log i) := by
  have hs : (fun j => if mode.present then
      Forward.TypedExecution.model.binary (some .fp32) .real .add
        (selectedValue Forward.TypedExecution.model Forward.TypedExecution.model.negInf K row causal input j)
        (maskedValue Forward.TypedExecution.model Forward.TypedExecution.model.negInf K row causal mask j)
      else selectedValue Forward.TypedExecution.model Forward.TypedExecution.model.negInf K row causal input j)=
        Forward.score K row causal mode xs masks := by
    funext j
    rw [selected_real]
    by_cases active : Forward.readActive K row causal j.val=Bool.true
    · cases mode <;> simp [Forward.MaskMode.present,Forward.TypedExecution.model,maskedValue,
        Forward.score,active,hx j active,hm,WithBot.realAdd]
    · cases mode <;> simp [Forward.MaskMode.present,Forward.TypedExecution.model,maskedValue,
        Forward.score,active,WithBot.realAdd]
  dsimp only [forwardValue]
  rw [hs,forwardTail_real hD,Forward.resultTile_extendedFormula,forward_formula K row hK hD]

theorem backwardValue_real {D : Nat} (K row : Nat) (log causal : Bool)
    (input saved : Fin D → WithBot ℝ) (g o : Fin D → ℝ)
    (hg : ∀ j, Forward.readActive K row causal j.val=Bool.true → input j=some (g j))
    (ho : ∀ j, Forward.readActive K row causal j.val=Bool.true → saved j=some (o j)) (i : Fin D) :
    backwardValue Forward.TypedExecution.model K row log causal input saved i=
      Cell.mk .real (some (gradient K row log causal g o i)) := by
  have hselected (x : Fin D → WithBot ℝ) (v : Fin D → ℝ)
      (h : ∀ j, Forward.readActive K row causal j.val=Bool.true → x j=some (v j)) :
      selectedValue Forward.TypedExecution.model (some 0) K row causal x=
        fun j => some (if Forward.readActive K row causal j.val then v j else 0) := by
    funext j
    rw [selected_real]
    by_cases ha : Forward.readActive K row causal j.val=Bool.true <;> simp [ha,h j]
  have hg':=hselected input g hg
  have ho':=hselected saved o ho
  dsimp only [backwardValue]
  rw [show Forward.TypedExecution.model.literal (some .fp32) .real 0=(some (0 : ℝ) : WithBot ℝ) from rfl]
  change Cell.mk .real (if causal then if i.val≤row then _ else some 0 else _)=_
  rw [hg',ho']
  cases log <;> cases causal
  all_goals
    simp [gradient,Backward.readActive,Forward.readActive,rowSum,Forward.TypedExecution.model,
      Tile.reduceSum,Tile.reduceSumDrop,TileShape.axisDim,TileShape.insertAxisIndex,
      WithBot.realSub,WithBot.realMul,WithBot.realExp,WithBot.sum_someTerm_eq_some]
    try rfl
  all_goals
    by_cases hr : i.val≤row
    · simp only [hr,if_true,show ¬(row < i.val) by omega,if_false]
      rfl
    · simp only [hr,if_false,show row < i.val by omega,if_true]

#axiomsClean forwardValue_real
#axiomsClean backwardValue_real
end
end VeriTile.Bench.Repairs.Ksoftmax

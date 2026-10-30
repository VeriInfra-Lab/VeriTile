import bench.tritonbench_g.ksoftmax_triton.Proofs.KernelWordCorrect

namespace VeriTile.Bench.Repairs.Ksoftmax.KernelLogBackward
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000

def probability (saved : WithBot ℝ) : ℝ :=
  match saved with | none => 0 | some value => Real.exp value

/-- Algebraic logarithmic JVP with exp(-infinity)=0. A derivative at an
infinite coordinate is not asserted. Future causal coordinates remain zero. -/
def formula {D : Nat} (K row : Nat) (causal : Bool) (gradient : Fin D → ℝ)
    (saved : Fin D → WithBot ℝ) (i : Fin D) : ℝ :=
  if causal && decide (row < i.val) then 0 else
    (if Forward.readActive K row causal i.val then gradient i else 0)-
    (if Forward.readActive K row causal i.val then probability (saved i) else 1)*
    ∑ j, if Forward.readActive K row causal j.val then gradient j else 0

theorem sum_some {D : Nat} (values : Fin D → ℝ) :
    rowSum Forward.TypedExecution.model (fun j => some (values j))=some (∑ j,values j) := by
  simp only [rowSum,Forward.TypedExecution.model,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.insertAxisIndex,WithBot.sum_someTerm_eq_some]
  rfl

theorem source_value {D : Nat} (K row : Nat) (causal : Bool)
    (input saved : Fin D → WithBot ℝ) (gradient : Fin D → ℝ)
    (binding : ∀ j, Forward.readActive K row causal j.val=Bool.true → input j=some (gradient j)) (i : Fin D) :
    backwardValue Forward.TypedExecution.model K row Bool.true causal input saved i=
      Cell.mk .real (some (formula K row causal gradient saved i)) := by
  have hg : selectedValue Forward.TypedExecution.model (some 0) K row causal input=
      fun j => some (if Forward.readActive K row causal j.val then gradient j else 0) := by
    funext j
    rw [selected_real]
    by_cases h : Forward.readActive K row causal j.val=Bool.true <;> simp [h,binding j]
  have ho : selectedValue Forward.TypedExecution.model (some 0) K row causal saved=
      fun j => if Forward.readActive K row causal j.val then saved j else some 0 := by
    funext j
    exact selected_real K row causal (some 0) saved j
  have exp_value : WithBot.realExp (if Forward.readActive K row causal i.val then saved i else some 0)=
      some (if Forward.readActive K row causal i.val then probability (saved i) else 1) := by
    split
    · cases saved i <;> rfl
    · simp [WithBot.realExp,Real.exp_zero]
  dsimp only [backwardValue]
  rw [show Forward.TypedExecution.model.literal (some .fp32) .real 0=(some (0 : ℝ) : WithBot ℝ) from rfl,hg,ho]
  simp only [if_true]
  change Cell.mk .real
    (if causal then if i.val≤row then
      WithBot.realSub (some (if Forward.readActive K row causal i.val then gradient i else 0))
        (WithBot.realMul (WithBot.realExp (if Forward.readActive K row causal i.val then saved i else some 0))
          (rowSum Forward.TypedExecution.model (fun j => some (if Forward.readActive K row causal j.val then gradient j else 0))))
      else some 0
    else WithBot.realSub (some (if Forward.readActive K row causal i.val then gradient i else 0))
      (WithBot.realMul (WithBot.realExp (if Forward.readActive K row causal i.val then saved i else some 0))
        (rowSum Forward.TypedExecution.model (fun j => some (if Forward.readActive K row causal j.val then gradient j else 0)))))=_
  rw [sum_some,exp_value]
  cases causal
  · rfl
  · by_cases h : i.val≤row
    · simp only [h,if_true,formula,Bool.true_and,decide_eq_true_eq,show ¬row < i.val by omega,if_false]
      rfl
    · simp only [h,if_false,formula,Bool.true_and,decide_eq_true_eq,show row < i.val by omega,if_true]
      rfl

#specNonCircular formula avoiding [forward,backward,backwardValue,runRows,Host.body]
#axiomsClean sum_some
#axiomsClean source_value
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelLogBackward

namespace VeriTile.Bench.Repairs.Ksoftmax.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
noncomputable section

def LogBackwardWordDomain (depth : Host.DepthLibrary) (grad saved : TensorMetadata) (causal : Bool)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved) (memory : Nat → Graph.Word) : Prop :=
  ∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))),
    Forward.readActive (columns saved) r causal i.val=Bool.true →
    KernelIdeal.finite (KernelTensor.values memory grad gradView (Native.flat (rows saved) (columns saved) b r i)) ∧
    KernelIdeal.finiteOrNegInf (KernelTensor.values memory saved savedView (Native.flat (rows saved) (columns saved) b r i))

def tensorLogBackwardFormula (D : Nat) (grad saved : TensorMetadata) (causal : Bool)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (batch row : Nat) (i : Fin D) : ℝ :=
  KernelLogBackward.formula (columns saved) row causal
    (fun j => (KernelIdeal.decode (KernelTensor.values memory grad gradView
      (Native.flat (rows saved) (columns saved) batch row j))).getD 0)
    (fun j => KernelIdeal.decode (KernelTensor.values memory saved savedView
      (Native.flat (rows saved) (columns saved) batch row j))) i

specification log_backward_kernel_real_words (depth : Host.DepthLibrary) (dst grad saved : TensorMetadata)
    (causal : Bool) (admitted : BackwardContract dst grad saved)
    (gradView : KernelTensor.View grad) (savedView : KernelTensor.View saved)
    (memory : Nat → Graph.Word) (s : State (WithBot ℝ))
    (domain : LogBackwardWordDomain depth grad saved causal gradView savedView memory) :
    Spec.Real (∃ final,
      KernelExecution.exec Forward.TypedExecution.model depth (.backward Bool.true causal) saved
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s)=some final ∧
      (∀ (b : Fin (batches saved)) (r : Fin (rows saved)) (i : Fin (depth.depth (columns saved))), i.val<columns saved →
        final.mem "dst" (Native.flat (rows saved) (columns saved) b r i.val)=Cell.mk .real
          (some (tensorLogBackwardFormula _ grad saved causal gradView savedView memory b r i))) ∧
      (∀ (region : RegionName) j, region≠"dst" ∨ saved.numel≤j → final.mem region j=
        (KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s).mem region j)) := by
  let initial := KernelTensor.idealInitial KernelIdeal.decode dst grad saved memory s
  refine ⟨Host.body Forward.TypedExecution.model depth (.backward Bool.true causal) saved initial,
    KernelExecution.success _ _ _ _ _,?_,?_⟩
  · intro b r i hi
    rw [Host.body_value _ _ _ _ admitted.2.2.2.1 initial b r i hi]
    simp only [tensorLogBackwardFormula,
      KernelTensor.values_words memory grad gradView admitted.2.1,
      KernelTensor.values_words memory saved savedView admitted.2.2.1]
    change backwardValue Forward.TypedExecution.model (columns saved) r Bool.true causal
      (fun j => (initial.mem "left" (Native.flat (rows saved) (columns saved) b r j.val)).read .real)
      (fun j => (initial.mem "right" (Native.flat (rows saved) (columns saved) b r j.val)).read .real) i=_
    simp only [initial,KernelTensor.ideal_left,KernelTensor.ideal_right]
    apply KernelLogBackward.source_value
    intro j hj
    apply KernelIdeal.decode_finite_value
    have h := (domain b r j hj).1
    simpa only [KernelTensor.values_words memory grad gradView admitted.2.1] using h
  · exact Host.body_frame _ _ _ _ admitted.2.2.2.1 initial

#specNonCircular LogBackwardWordDomain avoiding [forward,backward,runRows,Host.body,KernelExecution.exec]
#specNonCircular tensorLogBackwardFormula avoiding [forward,backward,runRows,Host.body,KernelExecution.exec]
#axiomsClean log_backward_kernel_real_words
end
end VeriTile.Bench.Repairs.Ksoftmax.Correct

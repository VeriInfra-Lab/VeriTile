import bench.tritonbench_g.ksoftmax_triton.Proofs.ValueMath

/-! Independent finite-or-negative-infinity mathematics. A row must contain
at least one finite effective score. This is a mathematical domain, never an
additional Python guard; NaN/+infinity/all-bottom rows retain native semantics.
-/
namespace VeriTile.Bench.Repairs.Ksoftmax.KernelExtended
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 2000000

def scores {D : Nat} (K row : Nat) (causal : Bool) (mode : Forward.MaskMode)
    (xs masks : Fin D → WithBot ℝ) (i : Fin D) : WithBot ℝ :=
  if Forward.readActive K row causal i.val then
    if mode.present then Option.map₂ (· + ·) (xs i) (masks i) else xs i
  else none

def valid {D : Nat} (values : Fin D → WithBot ℝ) : Prop :=
  ∃ i a, values i=some a

/-- Unshifted partition over precisely the finite effective scores. -/
def formula {D : Nat} (values : Fin D → WithBot ℝ) (log : Bool) (i : Fin D) : WithBot ℝ :=
  softmax (fun j => (values j).isSome) (fun j => (values j).getD 0) log i

theorem maximum_finite {D : Nat} (values : Fin D → WithBot ℝ) (domain : valid values) :
    ∃ maximum : ℝ, Forward.rowMaximum values=some maximum := by
  cases hm : Forward.rowMaximum values with
  | bot =>
    obtain ⟨i,a,hi⟩ := domain
    have every := (Finset.sup_eq_bot_iff values Finset.univ).mp hm
    have bad := every i (Finset.mem_univ i)
    rw [hi] at bad
    contradiction
  | coe maximum => exact ⟨maximum,rfl⟩

theorem tail_formula {D : Nat} (hD : 0<D) (values : Fin D → WithBot ℝ)
    (domain : valid values) (log : Bool) (i : Fin D) :
    forwardTail Forward.TypedExecution.model log values i=Cell.mk .real (formula values log i) := by
  obtain ⟨maximum,hm⟩ := maximum_finite values domain
  let active := fun j : Fin D => (values j).isSome
  let finite := fun j : Fin D => (values j).getD 0
  have ha : ∃ j, active j=Bool.true := by
    obtain ⟨j,a,hj⟩ := domain
    exact ⟨j,by simp [active,hj]⟩
  have num (j : Fin D) :
      WithBot.realExp (WithBot.realSub (values j) (some maximum))=
        some (if active j then Real.exp (finite j-maximum) else 0) := by
    cases hv : values j <;> simp only [active,finite,hv] <;> rfl
  have den : rowSum Forward.TypedExecution.model
      (fun j => WithBot.realExp (WithBot.realSub (values j) (some maximum)))=
      some (shiftedPartition active finite maximum) := by
    simp_rw [num]
    simp only [rowSum,Forward.TypedExecution.model,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.insertAxisIndex,WithBot.sum_someTerm_eq_some]
    rfl
  unfold forwardTail
  rw [rowMax_real hD,hm]
  change Cell.mk .real
    (if log then WithBot.realSub (WithBot.realSub (values i) (some maximum))
      (WithBot.realLog (rowSum Forward.TypedExecution.model
        (fun j => WithBot.realExp (WithBot.realSub (values j) (some maximum)))))
    else WithBot.realDiv (WithBot.realExp (WithBot.realSub (values i) (some maximum)))
      (rowSum Forward.TypedExecution.model
        (fun j => WithBot.realExp (WithBot.realSub (values j) (some maximum)))))=_
  rw [den,num i]
  change _=Cell.mk .real (softmax active finite log i)
  cases hv : values i using Option.rec with
  | none => cases log <;> simp [softmax,active,finite,hv,WithBot.realSub,WithBot.realLog,WithBot.realDiv]
  | some v =>
    have hi : active i=Bool.true := by simp [active,hv]
    have hf : finite i=v := by simp [finite,hv]
    cases log
    · simp only [softmax,Bool.false_eq_true,if_false,hi,if_true,WithBot.realDiv]
      change Cell.mk .real (some (Real.exp (finite i-maximum)/shiftedPartition active finite maximum))=
        Cell.mk .real (some (Real.exp (finite i)/partition active finite))
      rw [stable_plain]
    · simp only [softmax,hi,hf,if_true,WithBot.realSub,WithBot.realLog]
      change Cell.mk .real (some (v-maximum-Real.log (shiftedPartition active finite maximum)))=
        Cell.mk .real (some (v-Real.log (partition active finite)))
      rw [←hf,stable_log active finite maximum i ha]

theorem forward_value {D : Nat} (K row : Nat) (hD : 0<D) (log causal : Bool)
    (mode : Forward.MaskMode) (xs masks : Fin D → WithBot ℝ)
    (domain : valid (scores K row causal mode xs masks)) (i : Fin D) :
    forwardValue Forward.TypedExecution.model K row log causal mode xs masks i=
      Cell.mk .real (formula (scores K row causal mode xs masks) log i) := by
  have score_eq : (fun j => if mode.present then
      Forward.TypedExecution.model.binary (some .fp32) .real .add
        (selectedValue Forward.TypedExecution.model Forward.TypedExecution.model.negInf K row causal xs j)
        (maskedValue Forward.TypedExecution.model Forward.TypedExecution.model.negInf K row causal masks j)
      else selectedValue Forward.TypedExecution.model Forward.TypedExecution.model.negInf K row causal xs j)=
      scores K row causal mode xs masks := by
    funext j
    rw [selected_real]
    by_cases h : Forward.readActive K row causal j.val=Bool.true
    · cases mode <;> simp [scores,h,maskedValue,Forward.TypedExecution.model,Forward.MaskMode.present,WithBot.realAdd]
    · cases mode <;> simp [scores,h,maskedValue,Forward.TypedExecution.model,Forward.MaskMode.present,WithBot.realAdd]
  dsimp only [forwardValue]
  rw [score_eq]
  exact tail_formula hD _ domain log i

theorem bottom_output {D : Nat} (values : Fin D → WithBot ℝ) (log : Bool)
    (i : Fin D) (masked : values i=none) :
    formula values log i=if log then none else some 0 := by
  cases log <;> simp only [formula,softmax,masked] <;> rfl

#specNonCircular scores avoiding [forward,backward,runRows]
#specNonCircular formula avoiding [forwardTail,forwardValue,forward,backward,runRows]
#axiomsClean maximum_finite
#axiomsClean tail_formula
#axiomsClean forward_value
#axiomsClean bottom_output
end
end VeriTile.Bench.Repairs.Ksoftmax.KernelExtended

import bench.tritonbench_g.ksoftmax_triton.Proofs.Source
import bench.tritonbench_g.ksoftmax_triton.Proofs.Mathematical_ForwardMath
import Mathlib.Analysis.SpecialFunctions.Log.Basic

namespace VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped BigOperators
noncomputable section

/-- Mathematical Jacobian-vector product; future causal coordinates are absent
from the forward map and therefore have derivative zero. -/
def gradient {D : Nat} (K row : Nat) (log causal : Bool)
    (g o : Fin D → ℝ) (i : Fin D) : ℝ :=
  if causal && decide (row < i.val) then 0
  else if log then
    (if Backward.readActive K row causal i.val then g i else 0) -
      Real.exp (if Backward.readActive K row causal i.val then o i else 0) *
        ∑ j, if Backward.readActive K row causal j.val then g j else 0
  else
    (if Backward.readActive K row causal i.val then o i else 0) *
      ((if Backward.readActive K row causal i.val then g i else 0) -
        ∑ j, (if Backward.readActive K row causal j.val then g j else 0) *
             (if Backward.readActive K row causal j.val then o j else 0))

theorem gradient_source_formula {D : Nat} (K row : Nat) (log causal : Bool)
    (g o : Fin D → ℝ) (i : Fin D) :
    gradient K row log causal g o i =
      if causal && decide (row < i.val) then 0 else Backward.formula K row log causal g o i := by
  simp only [gradient,Backward.formula,Backward.masked]

theorem gradient_future {D : Nat} (K row : Nat) (log : Bool)
    (g o : Fin D → ℝ) (i : Fin D) (future : row < i.val) :
    gradient K row log Bool.true g o i=0 := by simp [gradient,future]

theorem gradient_active {D : Nat} (K row : Nat) (log causal : Bool)
    (g o : Fin D → ℝ) (i : Fin D) (active : Backward.readActive K row causal i.val=Bool.true) :
    gradient K row log causal g o i =
      if log then g i-Real.exp (o i)*∑ j, Backward.masked K row causal g j
      else o i*(g i-∑ j, Backward.masked K row causal g j*Backward.masked K row causal o j) := by
  cases causal <;> simp_all [gradient,Backward.readActive,Backward.masked]

/-- Independent unshifted partition, separate from the source's maximum. -/
def partition {D : Nat} (active : Fin D → Bool) (score : Fin D → ℝ) : ℝ :=
  ∑ j, if active j then Real.exp (score j) else 0

def shiftedPartition {D : Nat} (active : Fin D → Bool) (score : Fin D → ℝ) (maximum : ℝ) : ℝ :=
  ∑ j, if active j then Real.exp (score j-maximum) else 0

def softmax {D : Nat} (active : Fin D → Bool) (score : Fin D → ℝ) (log : Bool) (i : Fin D) : WithBot ℝ :=
  if active i then
    some (if log then score i-Real.log (partition active score)
      else Real.exp (score i)/partition active score)
  else if log then none else some 0

theorem shifted_partition {D : Nat} (active : Fin D → Bool) (score : Fin D → ℝ) (maximum : ℝ) :
    shiftedPartition active score maximum=partition active score/Real.exp maximum := by
  simp only [shiftedPartition,partition,Finset.sum_div]
  apply Finset.sum_congr rfl
  intro i _
  split <;> simp_all [Real.exp_sub]

theorem partition_positive {D : Nat} (active : Fin D → Bool) (score : Fin D → ℝ)
    (nonempty : ∃ i, active i=Bool.true) : 0<partition active score := by
  obtain ⟨i,hi⟩:=nonempty
  unfold partition
  apply Finset.sum_pos' (fun j _ => by split <;> positivity)
  exact ⟨i,Finset.mem_univ i,by simp [hi,Real.exp_pos]⟩

theorem stable_plain {D : Nat} (active : Fin D → Bool) (score : Fin D → ℝ)
    (maximum : ℝ) (i : Fin D) :
    Real.exp (score i-maximum)/shiftedPartition active score maximum=
      Real.exp (score i)/partition active score := by
  rw [Real.exp_sub,shifted_partition]
  field_simp [Real.exp_ne_zero]

theorem stable_log {D : Nat} (active : Fin D → Bool) (score : Fin D → ℝ)
    (maximum : ℝ) (i : Fin D) (nonempty : ∃ j, active j=Bool.true) :
    score i-maximum-Real.log (shiftedPartition active score maximum)=
      score i-Real.log (partition active score) := by
  rw [shifted_partition,Real.log_div (ne_of_gt (partition_positive active score nonempty))
    (Real.exp_ne_zero maximum),Real.log_exp]
  ring

/-- Every positive row has an active first lane, including causal row zero. -/
theorem active_nonempty (K D row : Nat) (hK : 0<K) (hD : 0<D) (causal : Bool) :
    ∃ i : Fin D, Forward.readActive K row causal i.val=Bool.true := by
  refine ⟨⟨0,hD⟩,?_⟩
  cases causal <;> simp [Forward.readActive,hK]

/-- Convert the historical extended source formula to the independent
unshifted mathematical softmax; unlike its real-memory observer, this retains
negative infinity at causal log-output coordinates. -/
theorem forward_formula {D : Nat} (K row : Nat) (hK : 0<K) (hD : 0<D)
    (log causal : Bool) (mode : Forward.MaskMode) (xs masks : Fin D → ℝ) (i : Fin D) :
    Forward.extendedFormula K row log causal mode xs masks i =
      softmax (fun j => Forward.readActive K row causal j.val)
        (fun j => xs j + if mode.present then masks j else 0) log i := by
  let active := fun j : Fin D => Forward.readActive K row causal j.val
  let scores := fun j : Fin D => xs j+if mode.present then masks j else 0
  have ha : ∃ j, active j=Bool.true := active_nonempty K D row hK hD causal
  have hmax : ∃ maximum : ℝ,
      Forward.rowMaximum (Forward.score K row causal mode xs masks)=some maximum := by
    cases hm : Forward.rowMaximum (Forward.score K row causal mode xs masks) with
    | bot =>
      obtain ⟨j,hj⟩:=ha
      have each := (Finset.sup_eq_bot_iff (Forward.score K row causal mode xs masks) Finset.univ).mp hm
      have bad:=each j (Finset.mem_univ j)
      simp [Forward.score,active] at hj
      simp [Forward.score,hj] at bad
    | coe maximum => exact ⟨maximum,rfl⟩
  obtain ⟨maximum,hm⟩:=hmax
  have shift (j : Fin D) : Forward.shifted (Forward.score K row causal mode xs masks) j =
      if active j then some (scores j-maximum) else none := by
    unfold Forward.shifted
    rw [hm]
    by_cases hj : active j=Bool.true
    · simp only [Forward.score,show Forward.readActive K row causal j.val=Bool.true from hj,hj,if_true]
      rfl
    · have hj' : active j=Bool.false := Bool.eq_false_iff.mpr hj
      simp only [Forward.score,show Forward.readActive K row causal j.val=Bool.false from hj',
        Bool.false_eq_true,if_false,hj']
      rfl
  have num (j : Fin D) : Forward.numerator (Forward.score K row causal mode xs masks) j =
      if active j then Real.exp (scores j-maximum) else 0 := by
    unfold Forward.numerator
    rw [shift]
    cases active j <;> rfl
  have den : Forward.denominator (Forward.score K row causal mode xs masks)=
      shiftedPartition active scores maximum := by
    unfold Forward.denominator shiftedPartition
    apply Finset.sum_congr rfl
    intro j _
    exact num j
  change (if log then Option.map (fun z => z-Real.log (Forward.denominator
      (Forward.score K row causal mode xs masks)))
      (Forward.shifted (Forward.score K row causal mode xs masks) i)
    else some (Forward.numerator (Forward.score K row causal mode xs masks) i /
      Forward.denominator (Forward.score K row causal mode xs masks))) = softmax active scores log i
  rw [shift,num,den]
  by_cases hi : active i=Bool.true
  · cases log
    · simp only [Bool.false_eq_true,if_false,hi,if_true,softmax]
      exact congrArg (fun x : ℝ => (some x : WithBot ℝ)) (stable_plain active scores maximum i)
    · simp only [if_true,hi,softmax,Option.map_some]
      exact congrArg (fun x : ℝ => (some x : WithBot ℝ)) (stable_log active scores maximum i ha)
  · have hi' : active i=Bool.false := Bool.eq_false_iff.mpr hi
    cases log <;> simp only [softmax,hi',Bool.false_eq_true,if_false,
      if_true,Option.map_none,zero_div]

#axiomsClean forward_formula
#axiomsClean gradient_source_formula
#axiomsClean gradient_future
#axiomsClean gradient_active
#axiomsClean shifted_partition
#axiomsClean partition_positive
#axiomsClean stable_plain
#axiomsClean stable_log
end
end VeriTile.Bench.Repairs.Ksoftmax

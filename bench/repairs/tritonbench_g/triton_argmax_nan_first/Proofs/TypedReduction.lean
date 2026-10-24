import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.ProjectedKernels
import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.NanReduce

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReduction
open VeriTile.Triton
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section
set_option maxHeartbeats 2000000

/-- Exact identity of source cast and where forms for a boolean flag. -/
theorem flag_cast_where (b : Bool) :
    (if b then (1 : Int) else 0)=(if b then 1 else 0) ∧
      (BitVec.ofInt 32 (if b then (1 : Int) else 0)).toInt=(if b then 1 else 0) := by
  cases b <;> decide

theorem self_equal (x : Word) : ScalarBinding.eq32 x x= !MachineMax.isNaN .fp32 x := by
  cases h : MachineMax.isNaN .fp32 x <;> simp [ScalarBinding.eq32,h]

private theorem first_true_fold {α : Type} (flags : α → Bool) (indices : List α) (initial : α) :
    indices.foldl (fun best candidate =>
      if (if flags best then (1 : Int) else 0)<(if flags candidate then 1 else 0)
      then candidate else best) initial=
      if flags initial then initial else (indices.find? flags).getD initial := by
  induction indices generalizing initial with
  | nil => simp
  | cons i rest ih =>
    simp only [List.foldl_cons]
    rw [ih]
    cases hi : flags i <;> cases hb : flags initial <;> simp [hi,hb]

theorem flags_exact {n : Nat} (input : Fin n → Word) (tree : Tree n)
    (positive : 0<n) (complete : tree.Complete)
    (nan : ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true) :
    TileShape.leftmostIntArgMax (flag input)=(flagIndex input tree).val := by
  let chosen := flagIndex input tree
  have isNaN := (flag_exists input tree complete).mpr nan
  have found : (List.finRange n).find? (fun i => MachineMax.isNaN .fp32 (input i))=some chosen := by
    apply List.find?_eq_some_iff_getElem.mpr
    refine ⟨isNaN,chosen.val,by simpa using chosen.isLt,?_,?_⟩
    · simp only [List.getElem_finRange]
      exact Fin.ext rfl
    · intro j hj
      have bound : j<n := hj.trans chosen.isLt
      have no : MachineMax.isNaN .fp32 (input ⟨j,bound⟩)=Bool.false := by
        cases hn : MachineMax.isNaN .fp32 (input ⟨j,bound⟩)
        · rfl
        · have first := flag_first input tree complete nan ⟨j,bound⟩ hn
          change chosen.val ≤ j at first
          omega
      simpa using no
  unfold TileShape.leftmostIntArgMax
  rw [dif_pos positive]
  change ((List.finRange n).foldl (fun best candidate =>
    if (if MachineMax.isNaN .fp32 (input best) then (1:Int) else 0)<
      (if MachineMax.isNaN .fp32 (input candidate) then 1 else 0) then candidate else best) _).val=_
  rw [first_true_fold]
  by_cases first : MachineMax.isNaN .fp32 (input ⟨0,positive⟩)=Bool.true
  · rw [if_pos first]
    have h := flag_first input tree complete nan ⟨0,positive⟩ first
    exact Nat.eq_zero_of_le_zero h |>.symm
  · rw [if_neg first,found]
    rfl

def presence {n : Nat} (positive : 0<n) (input : Fin n → Word) : Int :=
  (Finset.univ : Finset (Fin n)).sup' ⟨⟨0,positive⟩,Finset.mem_univ _⟩ (flag input)

theorem presence_iff {n : Nat} (positive : 0<n) (input : Fin n → Word) :
    0<presence positive input ↔ ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true := by
  constructor
  · intro h
    by_contra absent
    have zero : ∀ i,flag input i≤0 := by
      intro i
      cases hn : MachineMax.isNaN .fp32 (input i)
      · simp [flag,hn]
      · exact False.elim (absent ⟨i,hn⟩)
    have bound : presence positive input≤0 := Finset.sup'_le _ _ (fun i _ => zero i)
    omega
  · rintro ⟨i,hi⟩
    have bound : flag input i≤presence positive input := Finset.le_sup' _ (Finset.mem_univ i)
    exact ((flag_positive input i).mpr hi).trans_le bound

/-- The exact integer evaluator and a canonical numeric indexed maximum.
Only sanitized non-NaN words reach the numeric primitive. -/
def selected {n : Nat} (positive : 0<n) (input : Fin n → Word) : Nat :=
  if 0<presence positive input then TileShape.leftmostIntArgMax (flag input)
  else (MachineMax.best .fp32 (clean input) positive).val

theorem numeric_correct {n : Nat} (positive : 0<n) (input : Fin n → Word) :
    IsArgmax (clean input) (MachineMax.best .fp32 (clean input) positive) := by
  constructor
  · intro i
    have h := MachineMax.best_maximal .fp32 (clean input) positive i
    change priority (clean input i) ≤ priority (clean input _)
    simpa [priority,clean_nonNaN] using h
  · intro i h
    apply MachineMax.best_leftmost .fp32 (clean input) positive i
    change priority (clean input i)=priority (clean input _) at h
    simpa [priority,clean_nonNaN] using h

theorem selected_eq {n : Nat} (positive : 0<n) (input : Fin n → Word)
    (tree : Tree n) (complete : tree.Complete) :
    selected positive input=(tree.pick greater equal input).val := by
  by_cases nan : ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true
  · have hp := (presence_iff positive input).mpr nan
    rw [selected,if_pos hp,flags_exact input tree positive complete nan]
    have present : hasNaN input tree = Bool.true := (hasNaN_iff input tree complete).mpr nan
    have same := nanReduce_eq input tree tree tree complete complete complete
    simp only [nanReduce,if_pos present] at same
    exact congrArg Fin.val same
  · have hp : ¬0<presence positive input := fun h => nan ((presence_iff positive input).mp h)
    have cleanSame : clean input=input := by
      funext i
      cases hn : MachineMax.isNaN .fp32 (input i)
      · simp [clean,hn]
      · exact False.elim (nan ⟨i,hn⟩)
    rw [selected,if_neg hp]
    have correct := numeric_correct positive input
    rw [cleanSame] at correct ⊢
    exact congrArg Fin.val (least_maximum_unique _ _ _ correct (pick_correct input tree complete))

#axiomsClean flag_cast_where
#axiomsClean flags_exact
#axiomsClean presence_iff
#axiomsClean selected_eq
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.TypedReduction

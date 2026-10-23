import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Target

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section

def flag {n : Nat} (input : Fin n → Word) (i : Fin n) : Int :=
  if MachineMax.isNaN .fp32 (input i) then 1 else 0

def clean {n : Nat} (input : Fin n → Word) (i : Fin n) : Word :=
  if MachineMax.isNaN .fp32 (input i) then 0xff800000 else input i

def flagIndex {n : Nat} (input : Fin n → Word) (tree : Tree n) : Fin n :=
  tree.pick orderedGt orderedEq (flag input)

def hasNaN {n : Nat} (input : Fin n → Word) (tree : Tree n) : Bool :=
  decide (0<flag input (flagIndex input tree))

/-- Exact source reduction sequence: test NaNs, integer first-index reduction,
integer presence reduction, sanitize only the numeric reduction, select the
index, then reread the original word. Independent complete trees may be used
at all three reduction sites. -/
def nanReduce {n : Nat} (input : Fin n → Word)
    (flags present numeric : Tree n) : Fin n :=
  if hasNaN input present then flagIndex input flags
  else numeric.pick ScalarBinding.gt32 ScalarBinding.eq32 (clean input)

theorem flag_positive {n : Nat} (input : Fin n → Word) (i : Fin n) :
    0<flag input i ↔ MachineMax.isNaN .fp32 (input i)=Bool.true := by
  cases h : MachineMax.isNaN .fp32 (input i) <;> simp [flag,h]

theorem flag_exists {n : Nat} (input : Fin n → Word) (tree : Tree n)
    (complete : tree.Complete) :
    MachineMax.isNaN .fp32 (input (flagIndex input tree))=Bool.true ↔
      ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true := by
  constructor
  · exact fun h => ⟨_,h⟩
  · rintro ⟨i,hi⟩
    apply (flag_positive input _).mp
    have bound := (mathematical_maximum (flag input) tree complete).1 i
    exact lt_of_lt_of_le ((flag_positive input i).mpr hi) bound

theorem hasNaN_iff {n : Nat} (input : Fin n → Word) (tree : Tree n)
    (complete : tree.Complete) :
    hasNaN input tree=Bool.true ↔ ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true := by
  simpa only [hasNaN,decide_eq_true_eq,flag_positive] using flag_exists input tree complete

theorem flag_first {n : Nat} (input : Fin n → Word) (tree : Tree n)
    (complete : tree.Complete) (nan : ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true)
    (i : Fin n) (hi : MachineMax.isNaN .fp32 (input i)=Bool.true) :
    (flagIndex input tree).val ≤ i.val := by
  have selected := (flag_exists input tree complete).mpr nan
  dsimp only [flagIndex] at selected
  apply (mathematical_maximum (flag input) tree complete).2 i
  simp only [flag,hi,selected,ite_true]

theorem clean_nonNaN {n : Nat} (input : Fin n → Word) (i : Fin n) :
    MachineMax.isNaN .fp32 (clean input i)=Bool.false := by
  cases h : MachineMax.isNaN .fp32 (input i)
  · simp [clean,h]
  · simp only [clean,h,ite_true]; decide

theorem clean_pick {n : Nat} (input : Fin n → Word) (tree : Tree n) :
    tree.pick ScalarBinding.gt32 ScalarBinding.eq32 (clean input)=
      tree.pick greater equal (clean input) := by
  induction tree with
  | leaf => rfl
  | fork a b ha hb =>
    simp only [Tree.pick,ha,hb,leftWins,greater,equal,clean_nonNaN,
      Bool.false_and,Bool.or_false]

theorem nanReduce_correct {n : Nat} (input : Fin n → Word)
    (flags present numeric : Tree n) (hf : flags.Complete) (hp : present.Complete)
    (hn : numeric.Complete) : IsArgmax input (nanReduce input flags present numeric) := by
  by_cases nan : ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true
  · have has := (hasNaN_iff input present hp).mpr nan
    have selected := (flag_exists input flags hf).mpr nan
    rw [nanReduce,if_pos has]
    constructor
    · intro i
      change priority (input i)≤priority (input (flagIndex input flags))
      rw [(priority_top _).mpr selected]
      exact le_top
    · intro i hi
      apply flag_first input flags hf nan i
      apply (priority_top _).mp
      exact hi.trans ((priority_top _).mpr selected)
  · have has : hasNaN input present=Bool.false := by
      cases h : hasNaN input present
      · rfl
      · exact False.elim (nan ((hasNaN_iff input present hp).mp h))
    have clean_eq : clean input=input := by
      funext i
      cases hi : MachineMax.isNaN .fp32 (input i)
      · simp [clean,hi]
      · exact False.elim (nan ⟨i,hi⟩)
    rw [nanReduce,if_neg (by simp [has]),clean_pick,clean_eq]
    exact pick_correct input numeric hn

theorem nanReduce_eq {n : Nat} (input : Fin n → Word)
    (flags present numeric : Tree n) (hf : flags.Complete) (hp : present.Complete)
    (hn : numeric.Complete) :
    nanReduce input flags present numeric=numeric.pick greater equal input :=
  least_maximum_unique _ _ _ (nanReduce_correct input flags present numeric hf hp hn)
    (pick_correct input numeric hn)

/-- Ordered running update commits a newly encountered NaN and retains an
earlier NaN. Equal numeric values retain the earlier index. -/
theorem update_exact (next current : Word) :
    ((!MachineMax.isNaN .fp32 current) &&
      (MachineMax.isNaN .fp32 next || ScalarBinding.gt32 next current))=greater next current := by
  cases hn : MachineMax.isNaN .fp32 next <;> cases hc : MachineMax.isNaN .fp32 current
  all_goals simp [greater,ScalarBinding.gt32,hn,hc]

#axiomsClean nanReduce_correct
#axiomsClean nanReduce_eq
#axiomsClean update_exact
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst

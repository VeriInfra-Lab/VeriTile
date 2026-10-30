import bench.tritonbench_g.triton_argmax.Proofs.NanReduce

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section

def padded (n start width : Nat) (input : Nat → Word) (lane : Fin width) : Word :=
  if start+lane.val<n then input (start+lane.val) else 0xff800000

def selected (n start width : Nat) (input : Nat → Word) (tree : Tree width) : Fin width :=
  tree.pick greater equal (padded n start width input)

theorem selected_valid (n start width : Nat) (input : Nat → Word) (tree : Tree width)
    (complete : tree.Complete) (active : start<n) (positive : 0<width) :
    start+(selected n start width input tree).val<n := by
  let chosen := selected n start width input tree
  have h := pick_correct (padded n start width input) tree complete
  by_contra invalid
  change ¬start+chosen.val<n at invalid
  have first : start+(⟨0,positive⟩ : Fin width).val<n := by simpa using active
  have bound := h.1 ⟨0,positive⟩
  change priority (padded n start width input ⟨0,positive⟩) ≤
    priority (padded n start width input chosen) at bound
  simp only [padded,if_pos first,if_neg invalid,Nat.add_zero] at bound
  have tied : priority (padded n start width input ⟨0,positive⟩)=
      priority (padded n start width input chosen) := by
    simp only [padded,if_pos first,if_neg invalid,Nat.add_zero]
    exact le_antisymm bound (padding_lower _)
  have minimal := h.2 ⟨0,positive⟩ tied
  change chosen.val ≤ 0 at minimal
  omega

theorem selected_maximal (n start width : Nat) (input : Nat → Word) (tree : Tree width)
    (complete : tree.Complete) (active : start<n) (positive : 0<width)
    (lane : Fin width) (valid : start+lane.val<n) :
    priority (input (start+lane.val)) ≤ priority (input (start+(selected n start width input tree).val)) ∧
    (priority (input (start+lane.val))=priority (input (start+(selected n start width input tree).val)) →
      (selected n start width input tree).val ≤ lane.val) := by
  have selectedValid := selected_valid n start width input tree complete active positive
  dsimp only [selected] at selectedValid ⊢
  have h := pick_correct (padded n start width input) tree complete
  constructor
  · simpa only [padded,if_pos valid,if_pos selectedValid] using h.1 lane
  · intro equal
    apply h.2 lane
    simpa only [padded,if_pos valid,if_pos selectedValid] using equal

def localPair (n start width : Nat) (input : Nat → Word) (tree : Tree width) : Word × Nat :=
  let lane := selected n start width input tree
  (padded n start width input lane,start+lane.val)

theorem local_facts (n start width : Nat) (input : Nat → Word) (tree : Tree width)
    (positive : 0<width) (complete : tree.Complete) (active : start<n) :
    start ≤ (localPair n start width input tree).2 ∧
    (localPair n start width input tree).2<start+width ∧
    (localPair n start width input tree).2<n ∧
    (localPair n start width input tree).1=input (localPair n start width input tree).2 ∧
    (∀ i,start ≤ i → i<start+width → i<n →
      priority (input i) ≤ priority (localPair n start width input tree).1 ∧
      (priority (input i)=priority (localPair n start width input tree).1 →
        (localPair n start width input tree).2 ≤ i)) := by
  have valid := selected_valid n start width input tree complete active positive
  have lane := (selected n start width input tree).isLt
  refine ⟨by dsimp [localPair]; omega,by dsimp [localPair]; omega,valid,?_,?_⟩
  · simp only [localPair,padded,if_pos valid]
  · intro i lower upper bound
    have h := selected_maximal n start width input tree complete active positive
      ⟨i-start,by omega⟩ (by change start+(i-start)<n; omega)
    have decomp : start+(i-start)=i := by omega
    change priority (input (start+(i-start))) ≤ _ ∧ _ at h
    rw [decomp] at h
    dsimp only [localPair]
    simp only [padded,if_pos valid]
    refine ⟨h.1,?_⟩
    intro equal
    have le := h.2 equal
    change _ ≤ i-start at le
    omega

#axiomsClean selected_valid
#axiomsClean selected_maximal
#axiomsClean local_facts
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst

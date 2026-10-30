import bench.tritonbench_g.triton_argmax.Proofs.TritonArgmax_Math_Real

/-! The dimension kernel's strict-update, arbitrary multi-tile mathematical
execution. The target remains the least index of the original Real sequence. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Math
open Reduction
noncomputable section

def DimPrefix (n cur : Nat) (input : Nat → ℝ) (current : Value × Nat) : Prop :=
  (∀ i, i < cur → i < n → (input i : Value) ≤ current.1) ∧
  (∀ i, i < cur → i < n → (input i : Value)=current.1 → current.2 ≤ i) ∧
  ((current.1=⊥ ∧ current.2=0) ∨
    (current.2 < cur ∧ current.2 < n ∧ current.1=(input current.2 : Value)))

def localPair (n cur width : Nat) (input : Nat → ℝ) (tree : Tree width) : Value × Nat :=
  let i := blockSelected n cur width input tree
  (padded n cur width input i,cur+i.val)

def strictStep (n cur width : Nat) (input : Nat → ℝ) (tree : Tree width)
    (current : Value × Nat) : Value × Nat :=
  let next := localPair n cur width input tree
  if current.1 < next.1 then next else current

theorem prefix_zero (n : Nat) (input : Nat → ℝ) : DimPrefix n 0 input (⊥,0) := by
  simp [DimPrefix]

theorem local_facts (n cur width : Nat) (input : Nat → ℝ) (tree : Tree width)
    (positive : 0 < width) (complete : tree.Complete) (active : cur < n) :
    cur ≤ (localPair n cur width input tree).2 ∧
    (localPair n cur width input tree).2 < cur+width ∧
    (localPair n cur width input tree).2 < n ∧
    (localPair n cur width input tree).1=(input (localPair n cur width input tree).2 : Value) ∧
    (∀ i,cur ≤ i → i < cur+width → i < n →
      (input i : Value) ≤ (localPair n cur width input tree).1 ∧
      ((input i : Value)=(localPair n cur width input tree).1 →
        (localPair n cur width input tree).2 ≤ i)) := by
  have valid := block_valid n cur width input tree complete active positive
  have lane := (blockSelected n cur width input tree).isLt
  refine ⟨by dsimp [localPair]; omega,by dsimp [localPair]; omega,valid,?_,?_⟩
  · simp only [localPair,padded,if_pos valid]
  · intro i lower upper bound
    have h := block_maximal n cur width input tree complete active positive
      ⟨i-cur,by omega⟩ (by change cur+(i-cur) < n; omega)
    have decomp : cur+(i-cur)=i := by omega
    change input (cur+(i-cur)) ≤ _ ∧ _ at h
    rw [decomp] at h
    dsimp [localPair]
    simp only [padded,if_pos valid,WithBot.coe_le_coe,WithBot.coe_eq_coe]
    refine ⟨h.1,?_⟩
    intro equal
    have le := h.2 equal
    change _ ≤ i-cur at le
    omega

theorem strict_step_prefix (n cur width : Nat) (input : Nat → ℝ) (tree : Tree width)
    (positive : 0 < width) (complete : tree.Complete) (active : cur < n)
    (current : Value × Nat) (old : DimPrefix n cur input current) :
    DimPrefix n (cur+width) input (strictStep n cur width input tree current) := by
  obtain ⟨lower,upper,bound,concrete,segment⟩ := local_facts n cur width input tree positive complete active
  obtain ⟨oldMax,oldTie,oldValue⟩ := old
  dsimp only [strictStep]
  split
  · rename_i update
    refine ⟨?_,?_,Or.inr ⟨upper,bound,concrete⟩⟩
    · intro i hi hn
      by_cases before : i < cur
      · exact (oldMax i before hn).trans update.le
      · exact (segment i (by omega) hi hn).1
    · intro i hi hn equal
      by_cases before : i < cur
      · have lt := (oldMax i before hn).trans_lt update
        exact False.elim ((ne_of_lt lt) equal)
      · exact (segment i (by omega) hi hn).2 equal
  · rename_i noUpdate
    have nextLe : (localPair n cur width input tree).1 ≤ current.1 := le_of_not_gt noUpdate
    have oldConcrete : current.2 < cur ∧ current.2 < n ∧ current.1=(input current.2 : Value) := by
      rcases oldValue with ⟨bottom,_⟩ | real
      · rw [concrete,bottom] at nextLe
        exact False.elim ((WithBot.not_coe_le_bot _) nextLe)
      · exact real
    refine ⟨?_,?_,Or.inr ⟨by omega,oldConcrete.2⟩⟩
    · intro i hi hn
      by_cases before : i < cur
      · exact oldMax i before hn
      · exact ((segment i (by omega) hi hn).1).trans nextLe
    · intro i hi hn equal
      by_cases before : i < cur
      · exact oldTie i before hn equal
      · omega

def dimensionLoop (n width : Nat) (positive : 0 < width) (input : Nat → ℝ)
    (tree : Nat → Tree width) (cur : Nat) (current : Value × Nat) : Value × Nat :=
  if h : cur < n then dimensionLoop n width positive input tree (cur+width)
    (strictStep n cur width input (tree cur) current) else current
termination_by n-cur
decreasing_by omega

theorem dimension_loop_prefix (n width : Nat) (positive : 0 < width) (input : Nat → ℝ)
    (tree : Nat → Tree width) (complete : ∀ cur,cur < n → (tree cur).Complete)
    (cur : Nat) (current : Value × Nat) (old : DimPrefix n cur input current) :
    DimPrefix n n input (dimensionLoop n width positive input tree cur current) := by
  by_cases active : cur < n
  · have next := strict_step_prefix n cur width input (tree cur) positive (complete cur active) active current old
    have rest := dimension_loop_prefix n width positive input tree complete (cur+width)
      (strictStep n cur width input (tree cur) current) next
    rw [dimensionLoop,dif_pos active]
    exact rest
  · rw [dimensionLoop,dif_neg active]
    obtain ⟨maximum,tie,real⟩ := old
    refine ⟨fun i _ hn => maximum i (by omega) hn,fun i _ hn => tie i (by omega) hn,?_⟩
    rcases real with bottom | real
    · exact Or.inl bottom
    · exact Or.inr ⟨real.2.1,real.2⟩
termination_by n-cur
decreasing_by omega

def dimensionIndex (n width : Nat) (positive : 0 < width) (input : Nat → ℝ)
    (tree : Nat → Tree width) : Nat :=
  (dimensionLoop n width positive input tree 0 (⊥,0)).2

theorem dimension_mathematical (n width : Nat) (nonempty : 0 < n) (positive : 0 < width)
    (input : Nat → ℝ) (tree : Nat → Tree width)
    (complete : ∀ cur,cur < n → (tree cur).Complete) :
    ∃ bound : dimensionIndex n width positive input tree < n,
      IsLeastMaximum (fun i : Fin n => input i.val) ⟨dimensionIndex n width positive input tree,bound⟩ := by
  obtain ⟨maximum,tie,value⟩ := dimension_loop_prefix n width positive input tree complete 0 (⊥,0) (prefix_zero n input)
  have real : (dimensionLoop n width positive input tree 0 (⊥,0)).2 < n ∧
      (dimensionLoop n width positive input tree 0 (⊥,0)).1=
        (input (dimensionLoop n width positive input tree 0 (⊥,0)).2 : Value) := by
    rcases value with ⟨bottom,_⟩ | ⟨_,bound,concrete⟩
    · have impossible := maximum 0 nonempty nonempty
      rw [bottom] at impossible
      exact False.elim ((WithBot.not_coe_le_bot _) impossible)
    · exact ⟨bound,concrete⟩
  refine ⟨real.1,?_,?_⟩
  · intro i
    have h := maximum i.val i.isLt i.isLt
    rw [real.2] at h
    exact WithBot.coe_le_coe.mp h
  · intro i equal
    apply tie i.val i.isLt i.isLt
    rw [real.2]
    exact congrArg (fun r : ℝ => (r : Value)) equal

#axiomsClean strict_step_prefix
#axiomsClean dimension_loop_prefix
#axiomsClean dimension_mathematical
end
end VeriTile.Bench.Repairs.TritonArgmax.Math

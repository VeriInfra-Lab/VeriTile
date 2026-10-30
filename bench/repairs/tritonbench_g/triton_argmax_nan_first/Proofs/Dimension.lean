import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Padded
import bench.repairs.tritonbench_g.triton_argmax.Proofs.LoopLink

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 2000000

def fp32 : MidFormat := ⟨"fp32",4,by decide⟩
def fp32Scalars : Scalars fp32 where
  gt32 := greater
  eq32 := equal
  negInf32 := 0xff800000
  storeCast := id
  gtMid := greater
  eqMid := equal
  negInfMid := 0xff800000

def Prefix (n cur : Nat) (input : Nat → Word) (current : Word × Nat) : Prop :=
  (∀ i,i<cur → i<n → priority (input i) ≤ priority current.1) ∧
  (∀ i,i<cur → i<n → priority (input i)=priority current.1 → current.2 ≤ i) ∧
  (current=(0xff800000,0) ∨
    current.2<cur ∧ current.2<n ∧ current.1=input current.2)

def strictStep (n cur width : Nat) (input : Nat → Word) (tree : Tree width)
    (current : Word × Nat) : Word × Nat :=
  let next := localPair n cur width input tree
  if greater next.1 current.1 then next else current

theorem prefix_zero (n : Nat) (input : Nat → Word) : Prefix n 0 input (0xff800000,0) := by
  simp [Prefix]

theorem step_prefix (n cur width : Nat) (input : Nat → Word) (tree : Tree width)
    (positive : 0<width) (complete : tree.Complete) (active : cur<n)
    (current : Word × Nat) (old : Prefix n cur input current) :
    Prefix n (cur+width) input (strictStep n cur width input tree current) := by
  obtain ⟨lower,upper,bound,concrete,segment⟩ := local_facts n cur width input tree positive complete active
  obtain ⟨oldMax,oldTie,oldValue⟩ := old
  dsimp only [strictStep]
  rw [greater_priority]
  simp only [orderedGt,decide_eq_true_eq]
  split
  · rename_i update
    refine ⟨?_,?_,Or.inr ⟨upper,bound,concrete⟩⟩
    · intro i hi hn
      by_cases before : i<cur
      · exact (oldMax i before hn).trans update.le
      · exact (segment i (by omega) hi hn).1
    · intro i hi hn equal
      by_cases before : i<cur
      · exact False.elim ((ne_of_lt ((oldMax i before hn).trans_lt update)) equal)
      · exact (segment i (by omega) hi hn).2 equal
  · rename_i noUpdate
    have nextLe : priority (localPair n cur width input tree).1 ≤ priority current.1 :=
      le_of_not_gt noUpdate
    refine ⟨?_,?_,?_⟩
    · intro i hi hn
      by_cases before : i<cur
      · exact oldMax i before hn
      · exact ((segment i (by omega) hi hn).1).trans nextLe
    · intro i hi hn equal
      by_cases before : i<cur
      · exact oldTie i before hn equal
      · rcases oldValue with same | oldValue
        · rw [same]; exact Nat.zero_le i
        · omega
    · rcases oldValue with same | oldValue
      · exact Or.inl same
      · exact Or.inr ⟨by omega,oldValue.2⟩

def loop (n width : Nat) (positive : 0<width) (input : Nat → Word)
    (tree : Nat → Tree width) (cur : Nat) (current : Word × Nat) : Word × Nat :=
  if h : cur<n then loop n width positive input tree (cur+width)
    (strictStep n cur width input (tree cur) current) else current
termination_by n-cur
decreasing_by omega

theorem loop_prefix (n width : Nat) (positive : 0<width) (input : Nat → Word)
    (tree : Nat → Tree width) (complete : ∀ cur,cur<n → (tree cur).Complete)
    (cur : Nat) (current : Word × Nat) (old : Prefix n cur input current) :
    Prefix n n input (loop n width positive input tree cur current) := by
  by_cases active : cur<n
  · rw [loop,dif_pos active]
    apply loop_prefix n width positive input tree complete
    exact step_prefix n cur width input (tree cur) positive (complete cur active) active current old
  · rw [loop,dif_neg active]
    obtain ⟨maximum,tie,value⟩ := old
    refine ⟨fun i _ hn => maximum i (by omega) hn,fun i _ hn => tie i (by omega) hn,?_⟩
    rcases value with same | value
    · exact Or.inl same
    · exact Or.inr ⟨value.2.1,value.2⟩
termination_by n-cur
decreasing_by omega

theorem loop_correct (n width : Nat) (nonempty : 0<n) (positive : 0<width)
    (input : Nat → Word) (tree : Nat → Tree width)
    (complete : ∀ cur,cur<n → (tree cur).Complete) :
    ∃ bound : (loop n width positive input tree 0 (0xff800000,0)).2<n,
      IsArgmax (fun i : Fin n => input i.val)
        ⟨(loop n width positive input tree 0 (0xff800000,0)).2,bound⟩ := by
  obtain ⟨maximum,tie,value⟩ := loop_prefix n width positive input tree complete 0 _ (prefix_zero n input)
  rcases value with same | ⟨_,bound,concrete⟩
  · refine ⟨by rw [same]; exact nonempty,?_,?_⟩
    · intro i
      have h := maximum i.val i.isLt i.isLt
      have first := maximum 0 nonempty nonempty
      change priority (input i.val) ≤ priority (input (loop n width positive input tree 0 (0xff800000,0)).2)
      rw [same] at h first ⊢
      dsimp only [Prod.fst,Prod.snd] at h first ⊢
      exact h.trans (padding_lower _)
    · intro i _
      change (loop n width positive input tree 0 (0xff800000,0)).2 ≤ i.val
      rw [same]
      exact Nat.zero_le _
  · refine ⟨bound,?_,?_⟩
    · intro i
      have h := maximum i.val i.isLt i.isLt
      rw [concrete] at h
      exact h
    · intro i equal
      apply tie i.val i.isLt i.isLt
      rw [concrete]
      exact equal

def rowInput (c : DimConfig) (input : Nat → Word) (row inner : Nat) : Nat → Word :=
  fun col => input (dimInputOffset c row col inner)

theorem local_source (c : DimConfig) (input : Nat → Word) (row inner tile : Nat)
    (active : row<c.outer) :
    dimLocal fp32Scalars c input row inner tile=
      localPair c.extent (tile*128) 128 (rowInput c input row inner) (c.tree row inner tile) := by
  have loaded : c.loaded input fp32Scalars.negInf32 row inner tile=
      padded c.extent (tile*128) 128 (rowInput c input row inner) := by
    funext lane
    simp [DimConfig.loaded,padded,active,rowInput,fp32Scalars,blockN]
  simp only [dimLocal,localPair,selected,fp32Scalars,blockN]
  change c.loaded input 0xff800000 row inner tile = _ at loaded
  rw [loaded]

theorem source_scan (c : DimConfig) (input : Nat → Word) (row inner count tile : Nat)
    (active : row<c.outer) (ends : tile+count=(c.extent+127)/128) (current : Word × Nat) :
    loop c.extent 128 (by omega) (rowInput c input row inner)
      (fun cur => c.tree row inner (cur/128)) (tile*128) current=
      LoopLink.scan (fun t a => dimStep fp32Scalars c input row inner a t) count tile current := by
  induction count generalizing tile current with
  | zero =>
    have done : ¬tile*128<c.extent := by rw [←LoopLink.ceil_tiles]; omega
    rw [loop,dif_neg done]
    rfl
  | succ count ih =>
    have more : tile*128<c.extent := by rw [←LoopLink.ceil_tiles]; omega
    rw [loop,dif_pos more]
    have next : tile*128+128=(tile+1)*128 := by omega
    have div : tile*128/128=tile := by omega
    rw [next,div]
    rw [ih (tile+1) (by omega)]
    change _ = LoopLink.scan _ count (tile+1) (dimStep fp32Scalars c input row inner current tile)
    congr 1
    unfold dimStep strictStep
    rw [local_source c input row inner tile active]
    rfl

/-- Every source tile is executed. The final logical index is the unique
independent first-NaN/first-maximum index, with no non-NaN input restriction. -/
theorem dimension_reference (c : DimConfig) (input : Nat → Word) (row inner : Nat)
    (active : row<c.outer) (nonempty : 0<c.extent)
    (complete : ∀ tile,(c.tree row inner tile).Complete) :
    dimIndex fp32Scalars c input row inner=
      (reference c.extent nonempty (fun i : Fin c.extent => rowInput c input row inner i.val)).val := by
  have scan := source_scan c input row inner ((c.extent+127)/128) 0 active (by omega) (0xff800000,0)
  simp only [Nat.zero_mul,LoopLink.scan_fold] at scan
  obtain ⟨bound,correct⟩ := loop_correct c.extent 128 nonempty (by omega)
    (rowInput c input row inner) (fun cur => c.tree row inner (cur/128)) (fun cur _ => complete _)
  have unique := least_maximum_unique _ _ _ correct
    (reference_spec c.extent nonempty (fun i : Fin c.extent => rowInput c input row inner i.val))
  have result := congrArg Fin.val unique
  dsimp only [Fin.val_mk] at result
  rw [scan] at result
  simpa only [dimIndex,dimTiles,fp32Scalars,blockN,List.range_eq_range',
    show c.extent+128-1=c.extent+127 by omega] using result

#axiomsClean loop_correct
#axiomsClean local_source
#axiomsClean source_scan
#axiomsClean dimension_reference
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst

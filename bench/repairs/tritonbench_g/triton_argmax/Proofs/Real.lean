import bench.repairs.tritonbench_g.triton_argmax.Proofs.Tree

/-! Independent least-logical-index Real target, with no byte encoding of
mathematical values. Padding alone inhabits the bottom element. -/
namespace VeriTile.Bench.Repairs.TritonArgmax.Math
open Reduction Source
noncomputable section

def chain {n : Nat} (head : Fin n) (tail : List (Fin n)) : Tree n :=
  tail.foldl (fun t i => Tree.fork t (.leaf i)) (.leaf head)

theorem fold_leaves {n : Nat} (t : Tree n) (tail : List (Fin n)) :
    (tail.foldl (fun t i => Tree.fork t (.leaf i)) t).leaves = t.leaves ++ tail := by
  induction tail generalizing t with
  | nil => simp
  | cons i l ih => simp [List.foldl_cons,ih,Tree.leaves,List.append_assoc]

def enumeration (n : Nat) (positive : 0 < n) : Tree n := chain ⟨0,positive⟩ (List.finRange n)

theorem enumeration_complete (n : Nat) (positive : 0 < n) : (enumeration n positive).Complete := by
  intro i
  simp [enumeration,chain,fold_leaves,Tree.leaves]

/-- The target independently quantifies over arbitrary Real logical values. -/
def argmax (n : Nat) (positive : 0 < n) (input : Fin n → ℝ) : Fin n :=
  (enumeration n positive).pick orderedGt orderedEq input

theorem argmax_spec (n : Nat) (positive : 0 < n) (input : Fin n → ℝ) :
    IsLeastMaximum input (argmax n positive input) :=
  mathematical_maximum input (enumeration n positive) (enumeration_complete n positive)

abbrev Value := WithBot ℝ

def padded (size start width : Nat) (input : Nat → ℝ) (lane : Fin width) : Value :=
  if start+lane.val < size then (input (start+lane.val) : Value) else ⊥

def blockSelected (size start width : Nat) (input : Nat → ℝ) (tree : Tree width) : Fin width :=
  tree.pick orderedGt orderedEq (padded size start width input)

theorem block_valid (size start width : Nat) (input : Nat → ℝ) (tree : Tree width)
    (complete : tree.Complete) (active : start < size) (positive : 0 < width) :
    start+(blockSelected size start width input tree).val < size := by
  let selected := blockSelected size start width input tree
  have maximum := mathematical_maximum (padded size start width input) tree complete
  have first := maximum.1 ⟨0,positive⟩
  change padded size start width input ⟨0,positive⟩ ≤ padded size start width input selected at first
  by_contra invalid
  change ¬ start+selected.val < size at invalid
  have firstActive : start+0 < size := by omega
  simp only [padded,if_pos firstActive,if_neg invalid] at first
  exact (WithBot.not_coe_le_bot _) first

theorem block_maximal (size start width : Nat) (input : Nat → ℝ) (tree : Tree width)
    (complete : tree.Complete) (active : start < size) (positive : 0 < width)
    (lane : Fin width) (valid : start+lane.val < size) :
    input (start+lane.val) ≤ input (start+(blockSelected size start width input tree).val) ∧
    (input (start+lane.val)=input (start+(blockSelected size start width input tree).val) →
      (blockSelected size start width input tree).val ≤ lane.val) := by
  have maximum := mathematical_maximum (padded size start width input) tree complete
  have selectedValid := block_valid size start width input tree complete active positive
  have comparison := maximum.1 lane
  change padded size start width input lane ≤
    padded size start width input (blockSelected size start width input tree) at comparison
  simp only [padded,if_pos valid,if_pos selectedValid,WithBot.coe_le_coe] at comparison
  refine ⟨comparison,?_⟩
  intro equal
  apply maximum.2 lane
  change padded size start width input lane =
    padded size start width input (blockSelected size start width input tree)
  simp [padded,valid,selectedValid,equal]

structure GlobalCoverage (c : GlobalConfig) : Prop where
  sizePositive : 0 < c.size
  blockPositive : 0 < c.block
  blocksPositive : 0 < c.blocks
  cover : c.size ≤ c.blocks*c.block
  eachActive : ∀ pid, pid < c.blocks → pid*c.block < c.size
  midFits : c.blocks ≤ c.midBlock
  firstComplete : ∀ pid, pid < c.blocks → (c.firstTree pid).Complete
  secondComplete : c.secondTree.Complete

def firstIndex (c : GlobalConfig) (input : Nat → ℝ) (pid : Nat) : Nat :=
  pid*c.block+(blockSelected c.size (pid*c.block) c.block input (c.firstTree pid)).val

def midValues (c : GlobalConfig) (input : Nat → ℝ) (lane : Fin c.midBlock) : Value :=
  if lane.val < c.blocks then (input (firstIndex c input lane.val) : Value) else ⊥

def globalSelected (c : GlobalConfig) (input : Nat → ℝ) : Fin c.midBlock :=
  c.secondTree.pick orderedGt orderedEq (midValues c input)

def globalIndex (c : GlobalConfig) (input : Nat → ℝ) : Nat := firstIndex c input (globalSelected c input).val

theorem mid_valid (c : GlobalConfig) (coverage : GlobalCoverage c) (input : Nat → ℝ) :
    (globalSelected c input).val < c.blocks := by
  have maximum := mathematical_maximum (midValues c input) c.secondTree coverage.secondComplete
  have first := maximum.1 ⟨0,coverage.blocksPositive.trans_le coverage.midFits⟩
  change midValues c input ⟨0,coverage.blocksPositive.trans_le coverage.midFits⟩ ≤
    midValues c input (globalSelected c input) at first
  by_contra invalid
  simp only [midValues,if_pos coverage.blocksPositive,if_neg invalid] at first
  exact (WithBot.not_coe_le_bot _) first

theorem global_valid (c : GlobalConfig) (coverage : GlobalCoverage c) (input : Nat → ℝ) :
    globalIndex c input < c.size := by
  exact block_valid c.size _ c.block input _
    (coverage.firstComplete _ (mid_valid c coverage input))
    (coverage.eachActive _ (mid_valid c coverage input)) coverage.blockPositive

theorem global_mathematical (c : GlobalConfig) (coverage : GlobalCoverage c) (input : Nat → ℝ) :
    IsLeastMaximum (fun i : Fin c.size => input i.val)
      ⟨globalIndex c input,global_valid c coverage input⟩ := by
  let chosen := globalSelected c input
  have chosenValid : chosen.val < c.blocks := mid_valid c coverage input
  have maximum := mathematical_maximum (midValues c input) c.secondTree coverage.secondComplete
  have compareBlocks : ∀ pid, pid < c.blocks →
      input (firstIndex c input pid) ≤ input (globalIndex c input) ∧
      (input (firstIndex c input pid)=input (globalIndex c input) → chosen.val ≤ pid) := by
    intro pid valid
    let lane : Fin c.midBlock := ⟨pid,valid.trans_le coverage.midFits⟩
    have h := maximum.1 lane
    change (if pid < c.blocks then (input (firstIndex c input pid) : Value) else ⊥) ≤
      (if chosen.val < c.blocks then (input (firstIndex c input chosen.val) : Value) else ⊥) at h
    simp only [midValues,if_pos valid,if_pos chosenValid] at h
    refine ⟨WithBot.coe_le_coe.mp h,?_⟩
    intro equal
    apply maximum.2 lane
    change (if pid < c.blocks then (input (firstIndex c input pid) : Value) else ⊥) =
      (if chosen.val < c.blocks then (input (firstIndex c input chosen.val) : Value) else ⊥)
    simp only [midValues,if_pos valid,if_pos chosenValid]
    exact congrArg (fun r : ℝ => (r : Value)) equal
  have localFacts : ∀ i : Fin c.size,
      ∃ pid : Nat, pid < c.blocks ∧ pid*c.block ≤ i.val ∧ i.val < (pid+1)*c.block ∧
      input i.val ≤ input (firstIndex c input pid) ∧
      (input i.val=input (firstIndex c input pid) → firstIndex c input pid ≤ i.val) := by
    intro i
    let pid := i.val/c.block
    have rem := Nat.mod_lt i.val coverage.blockPositive
    have decomposition := Nat.div_add_mod i.val c.block
    have pidBound : pid < c.blocks := by
      apply (Nat.div_lt_iff_lt_mul coverage.blockPositive).mpr
      exact i.isLt.trans_le coverage.cover
    have startBound := coverage.eachActive pid pidBound
    have decomp : pid*c.block+i.val%c.block=i.val := by
      dsimp [pid]
      simpa only [Nat.mul_comm] using decomposition
    have active : pid*c.block+i.val%c.block < c.size := by rw [decomp]; exact i.isLt
    have h := block_maximal c.size (pid*c.block) c.block input (c.firstTree pid)
      (coverage.firstComplete pid pidBound) startBound coverage.blockPositive ⟨i.val%c.block,rem⟩ active
    rw [decomp] at h
    refine ⟨pid,pidBound,?_,?_,h.1,?_⟩
    · omega
    · nlinarith
    · intro equal
      have le := h.2 equal
      change (blockSelected c.size (pid*c.block) c.block input (c.firstTree pid)).val ≤ i.val%c.block at le
      unfold firstIndex
      omega
  constructor
  · intro i
    obtain ⟨pid,valid,_,_,le,_⟩ := localFacts i
    exact le.trans (compareBlocks pid valid).1
  · intro i equal
    change input i.val=input (globalIndex c input) at equal
    obtain ⟨pid,valid,lower,upper,le,tie⟩ := localFacts i
    have between := (compareBlocks pid valid).1
    have eq' : input i.val=input (firstIndex c input pid) := le_antisymm le (by rw [equal]; exact between)
    have chosenLe := (compareBlocks pid valid).2 (eq'.symm.trans equal)
    by_cases same : chosen.val=pid
    · change firstIndex c input chosen.val ≤ i.val
      rw [same]
      exact tie eq'
    · have less : chosen.val < pid := by omega
      have laneBound := (blockSelected c.size (chosen.val*c.block) c.block input (c.firstTree chosen.val)).isLt
      change chosen.val*c.block+
        (blockSelected c.size (chosen.val*c.block) c.block input (c.firstTree chosen.val)).val ≤ i.val
      have : (chosen.val+1)*c.block ≤ pid*c.block := Nat.mul_le_mul_right c.block (by omega)
      nlinarith

#axiomsClean argmax_spec
#axiomsClean block_valid
#axiomsClean block_maximal
#axiomsClean global_mathematical
end
end VeriTile.Bench.Repairs.TritonArgmax.Math

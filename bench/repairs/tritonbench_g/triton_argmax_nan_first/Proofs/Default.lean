import bench.repairs.tritonbench_g.triton_argmax_nan_first.Proofs.Dimension

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Default
open VeriTile.Bench.Repairs.TritonArgmax Source Reduction
noncomputable section
set_option maxHeartbeats 2000000

def first (c : GlobalConfig) (input : Nat → Word) (pid : Nat) : Word × Nat :=
  localPair c.size (pid*c.block) c.block input (c.firstTree pid)

def mid (c : GlobalConfig) (input : Nat → Word) (lane : Fin c.midBlock) : Word :=
  if lane.val<c.blocks then (first c input lane.val).1 else 0xff800000

def chosen (c : GlobalConfig) (input : Nat → Word) : Fin c.midBlock :=
  c.secondTree.pick greater equal (mid c input)

theorem chosen_valid (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → Word) :
    (chosen c input).val<c.blocks := by
  have maximum := pick_correct (mid c input) c.secondTree coverage.secondComplete
  by_contra invalid
  have firstLane : 0<c.midBlock := coverage.blocksPositive.trans_le coverage.midFits
  have bound := maximum.1 ⟨0,firstLane⟩
  change priority (mid c input ⟨0,firstLane⟩) ≤ priority (mid c input (chosen c input)) at bound
  simp only [mid,if_pos coverage.blocksPositive,if_neg invalid] at bound
  have equal : priority (mid c input ⟨0,firstLane⟩)=priority (mid c input (chosen c input)) := by
    simp only [mid,if_pos coverage.blocksPositive,if_neg invalid]
    exact le_antisymm bound (padding_lower _)
  have firstIndex := maximum.2 ⟨0,firstLane⟩ equal
  change (chosen c input).val ≤ 0 at firstIndex
  have positive := coverage.blocksPositive
  omega

def index (c : GlobalConfig) (input : Nat → Word) : Nat :=
  (first c input (chosen c input).val).2

theorem first_facts (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → Word)
    (pid : Nat) (active : pid<c.blocks) :
    pid*c.block ≤ (first c input pid).2 ∧
    (first c input pid).2<pid*c.block+c.block ∧
    (first c input pid).2<c.size ∧ (first c input pid).1=input (first c input pid).2 ∧
    (∀ i,pid*c.block ≤ i → i<pid*c.block+c.block → i<c.size →
      priority (input i) ≤ priority (first c input pid).1 ∧
      (priority (input i)=priority (first c input pid).1 → (first c input pid).2 ≤ i)) :=
  local_facts c.size (pid*c.block) c.block input (c.firstTree pid) coverage.blockPositive
    (coverage.firstComplete pid active) (coverage.eachActive pid active)

theorem valid (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → Word) :
    index c input<c.size := (first_facts c coverage input _ (chosen_valid c coverage input)).2.2.1

theorem mathematical (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → Word) :
    IsArgmax (fun i : Fin c.size => input i.val) ⟨index c input,valid c coverage input⟩ := by
  have chosenValid := chosen_valid c coverage input
  have bestFacts := first_facts c coverage input _ chosenValid
  have maximum := pick_correct (mid c input) c.secondTree coverage.secondComplete
  have blockMaximum : ∀ pid,pid<c.blocks →
      priority (first c input pid).1 ≤ priority (input (index c input)) ∧
      (priority (first c input pid).1=priority (input (index c input)) → (chosen c input).val ≤ pid) := by
    intro pid active
    let lane : Fin c.midBlock := ⟨pid,active.trans_le coverage.midFits⟩
    have bound := maximum.1 lane
    change priority (mid c input lane) ≤ priority (mid c input (chosen c input)) at bound
    simp only [mid,lane,if_pos active,if_pos chosenValid] at bound
    rw [bestFacts.2.2.2.1] at bound
    refine ⟨bound,?_⟩
    intro equal
    apply maximum.2 lane
    change priority (mid c input lane)=priority (mid c input (chosen c input))
    simp only [mid,lane,if_pos active,if_pos chosenValid,bestFacts.2.2.2.1]
    exact equal
  have localMaximum : ∀ i : Fin c.size,∃ pid,pid<c.blocks ∧ pid*c.block ≤ i.val ∧
      priority (input i.val) ≤ priority (first c input pid).1 ∧
      (priority (input i.val)=priority (first c input pid).1 → (first c input pid).2 ≤ i.val) := by
    intro i
    let pid := i.val/c.block
    have remainder := Nat.mod_lt i.val coverage.blockPositive
    have decomposition : pid*c.block+i.val%c.block=i.val := by
      dsimp [pid]
      simpa only [Nat.mul_comm] using Nat.div_add_mod i.val c.block
    have active : pid<c.blocks := (Nat.div_lt_iff_lt_mul coverage.blockPositive).mpr
      (i.isLt.trans_le coverage.cover)
    have lower : pid*c.block ≤ i.val := by omega
    have upper : i.val<pid*c.block+c.block := by omega
    have facts := (first_facts c coverage input pid active).2.2.2.2 i.val lower upper i.isLt
    exact ⟨pid,active,lower,facts⟩
  constructor
  · intro i
    obtain ⟨pid,active,_,bound,_⟩ := localMaximum i
    exact bound.trans (blockMaximum pid active).1
  · intro i equal
    change priority (input i.val)=priority (input (index c input)) at equal
    obtain ⟨pid,active,lower,bound,tie⟩ := localMaximum i
    have blocks := blockMaximum pid active
    have eqLocal : priority (input i.val)=priority (first c input pid).1 :=
      le_antisymm bound (by rw [equal]; exact blocks.1)
    have chosenLe := blocks.2 (eqLocal.symm.trans equal)
    by_cases same : (chosen c input).val=pid
    · change (first c input (chosen c input).val).2 ≤ i.val
      rw [same]
      exact tie eqLocal
    · have before : (chosen c input).val+1 ≤ pid := by omega
      have times := Nat.mul_le_mul_right c.block before
      change (first c input (chosen c input).val).2 ≤ i.val
      nlinarith [bestFacts.2.1]

theorem source (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → Word) :
    Source.globalIndex fp32Scalars c input=index c input := by
  have firstValues : stage1Value fp32Scalars c input=fun pid => (first c input pid).1 := by
    funext pid
    rfl
  have selected : stage2Selected fp32Scalars c (stage1Value fp32Scalars c input)=chosen c input := by
    rw [firstValues]
    rfl
  unfold Source.globalIndex
  rw [selected]
  dsimp only
  rw [if_pos (chosen_valid c coverage input)]
  simp only [stage1Index,index,first,localPair,TritonArgmaxNanFirst.selected,GlobalConfig.selected,fp32Scalars,
    GlobalConfig.loaded,padded]
  exact Nat.add_comm _ _

theorem reference_index (c : GlobalConfig) (coverage : Math.GlobalCoverage c) (input : Nat → Word) :
    Source.globalIndex fp32Scalars c input=
      (reference c.size coverage.sizePositive (fun i : Fin c.size => input i.val)).val := by
  rw [source c coverage input]
  exact congrArg Fin.val (least_maximum_unique _ _ _ (mathematical c coverage input)
    (reference_spec c.size coverage.sizePositive (fun i => input i.val)))

#axiomsClean mathematical
#axiomsClean reference_index
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.Default

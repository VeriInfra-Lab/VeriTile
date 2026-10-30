import bench.repairs.tritonbench_g.triton_argmax.Proofs.Real
import bench.repairs.tritonbench_g.triton_argmax.Proofs.ScalarBinding

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst
open VeriTile.Bench.Repairs.TritonArgmax
open Source Reduction
open VeriTile.Bench.TritonBenchG.MatrixReduction
noncomputable section

/-- NaNs share one top priority; all other FP32 words use the normalized IEEE
order key. Thus NaN payload/sign are irrelevant, ±0 tie, and infinities remain
ordered. This input-only target does not mention a tile or compiler tree. -/
def priority (word : Word) : WithTop Nat :=
  if MachineMax.isNaN .fp32 word then ⊤ else (MachineMax.rank .fp32 word : WithTop Nat)

def IsArgmax {n : Nat} (input : Fin n → Word) (index : Fin n) : Prop :=
  IsLeastMaximum (fun i => priority (input i)) index

def reference (n : Nat) (positive : 0<n) (input : Fin n → Word) : Fin n :=
  (Math.enumeration n positive).pick orderedGt orderedEq (fun i => priority (input i))

theorem reference_spec (n : Nat) (positive : 0<n) (input : Fin n → Word) :
    IsArgmax input (reference n positive input) :=
  mathematical_maximum _ _ (Math.enumeration_complete n positive)

theorem priority_top (word : Word) :
    priority word=⊤ ↔ MachineMax.isNaN .fp32 word=Bool.true := by
  cases h : MachineMax.isNaN .fp32 word <;> simp [priority,h]

/-- When any NaN exists, the independently specified result is its least
logical index. The statement includes every FP32 NaN payload and sign. -/
theorem first_nan {n : Nat} (input : Fin n → Word) (index : Fin n)
    (correct : IsArgmax input index) (hasNaN : ∃ i,MachineMax.isNaN .fp32 (input i)=Bool.true) :
    MachineMax.isNaN .fp32 (input index)=Bool.true ∧
    ∀ i,MachineMax.isNaN .fp32 (input i)=Bool.true → index.val ≤ i.val := by
  obtain ⟨nan,hnan⟩ := hasNaN
  have htop : priority (input index)=⊤ := by
    have h := correct.1 nan
    dsimp only at h
    rw [(priority_top _).mpr hnan] at h
    exact top_le_iff.mp h
  refine ⟨(priority_top _).mp htop,?_⟩
  intro i hi
  exact correct.2 i ((priority_top _).mpr hi |>.trans htop.symm)

def greater (x y : Word) : Bool :=
  ScalarBinding.gt32 x y || (MachineMax.isNaN .fp32 x && !MachineMax.isNaN .fp32 y)

def equal (x y : Word) : Bool :=
  ScalarBinding.eq32 x y || (MachineMax.isNaN .fp32 x && MachineMax.isNaN .fp32 y)

/-- IEEE comparison and NaN tests are concrete modeling definitions, not
numerical approximation premises. -/
theorem greater_priority (x y : Word) : greater x y=orderedGt (priority x) (priority y) := by
  cases hx : MachineMax.isNaN .fp32 x <;> cases hy : MachineMax.isNaN .fp32 y
  all_goals simp [greater,ScalarBinding.gt32,priority,orderedGt,hx,hy]

theorem equal_priority (x y : Word) : equal x y=orderedEq (priority x) (priority y) := by
  cases hx : MachineMax.isNaN .fp32 x <;> cases hy : MachineMax.isNaN .fp32 y
  all_goals simp [equal,ScalarBinding.eq32,priority,orderedEq,hx,hy]

theorem pick_priority {n : Nat} (input : Fin n → Word) (tree : Tree n) :
    tree.pick greater equal input=tree.pick orderedGt orderedEq (fun i => priority (input i)) := by
  induction tree with
  | leaf => rfl
  | fork a b ha hb =>
    simp only [Tree.pick,ha,hb,leftWins,greater_priority,equal_priority]
    rfl

theorem pick_correct {n : Nat} (input : Fin n → Word) (tree : Tree n) (complete : tree.Complete) :
    IsArgmax input (tree.pick greater equal input) := by
  rw [pick_priority]
  exact mathematical_maximum _ _ complete

theorem pick_reference {n : Nat} (positive : 0<n) (input : Fin n → Word)
    (tree : Tree n) (complete : tree.Complete) :
    tree.pick greater equal input=reference n positive input :=
  least_maximum_unique _ _ _ (pick_correct input tree complete) (reference_spec n positive input)

theorem rank_lower (word : Word) (notNaN : MachineMax.isNaN .fp32 word=Bool.false) :
    8388607≤MachineMax.rank .fp32 word := by
  have bound : word.toNat<4294967296 := word.isLt
  simp only [MachineMax.isNaN,MachineMax.magnitude,MachineMax.width,MachineMax.fractionBits,
    MachineMax.exponentBits] at notNaN
  norm_num at notNaN
  simp only [MachineMax.rank,BinaryConversion.key32_toNat]
  split_ifs <;> omega

theorem padding_lower (word : Word) : priority 0xff800000≤priority word := by
  have pad : priority 0xff800000=(8388607 : Nat) := by decide
  rw [pad]
  cases h : MachineMax.isNaN .fp32 word
  · simpa [priority,h] using rank_lower word h
  · simp [priority,h]

#axiomsClean reference_spec
#axiomsClean first_nan
#axiomsClean greater_priority
#axiomsClean equal_priority
#axiomsClean pick_correct
#axiomsClean padding_lower
end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst

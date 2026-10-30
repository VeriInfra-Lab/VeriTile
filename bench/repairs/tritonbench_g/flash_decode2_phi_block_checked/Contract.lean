import bench.repairs.tritonbench_g.flash_decode2_phi_block_checked.Kernels
import bench.repairs.tritonbench_g.square_matrix.Contract
import bench.repairs.tritonbench_g.flash_decode2_phi.Contract
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Admission

namespace VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked
open FlashDecode2
abbrev Call := FlashDecode2Phi.Call
abbrev OriginalContract (c : Call) := Checked.InputContract c.val

/-- Sole additional premise, exactly on calls that submit the two-dimensional
kernel grid. Every original empty-input check and accepted case is retained. -/
structure InputContract (c : Call) (lengths : Nat → Int) : Prop extends toOriginalContract : OriginalContract c lengths where
  arange : Checked.active c.val → Checked.block c.val ≤ 1048576

def checkInput (c : Call) (lengths : Nat → Int) : Except String Unit :=
  if ¬Checked.metadataChecks c.val then .error "metadata"
  else if Checked.active c.val ∧ ¬Checked.block c.val ≤ 1048576 then .error "lane block"
  else Checked.checkInput c.val lengths

theorem exact_domain (c : Call) (lengths : Nat → Int) :
    InputContract c lengths ↔ OriginalContract c lengths ∧
      (Checked.active c.val → Checked.block c.val ≤ 1048576) :=
  ⟨fun h => ⟨h.toOriginalContract,h.arange⟩,fun h => ⟨h.1,h.2⟩⟩

theorem check_input_iff (c : Call) (lengths : Nat → Int) :
    checkInput c lengths=.ok () ↔ InputContract c lengths := by
  rw [exact_domain]
  unfold checkInput
  by_cases hm : Checked.metadataChecks c.val
  · by_cases hb : Checked.active c.val ∧ ¬Checked.block c.val ≤ 1048576
    · simp [hm,hb]
    · simp only [hm,not_true_eq_false,↓reduceIte,hb,Checked.check_input_iff]
      exact ⟨fun h => ⟨h,by tauto⟩,fun h => h.1⟩
  · simp only [hm,not_false_eq_true,↓reduceIte,Except.error.injEq,reduceCtorEq,false_iff,not_and]
    exact fun h => False.elim (hm h.metadata)

theorem empty_domain_unchanged (c : Call) (lengths : Nat → Int)
    (empty : Checked.batches c.val=0 ∨ Checked.heads c.val=0) :
    InputContract c lengths ↔ OriginalContract c lengths := by
  rw [exact_domain]
  have inactive : ¬Checked.active c.val := by unfold Checked.active;omega
  simp [inactive]

def checkArguments (m e l o : Checked.Argument) (q : Checked.SequenceBlock)
    (lengths : Nat → Int) : Except String Unit :=
  match m,e,l,o,q with
  | .tensor m,.tensor e,.tensor l,.tensor o,.integer q =>
    if 0 < q then checkInput ⟨⟨m,e,l,o,q.toNat,Bool.true⟩,rfl⟩ lengths else .error "sequence block"
  | _,_,_,_,_ => .error "argument type"

def AdmittedArguments (m e l o : Checked.Argument) (q : Checked.SequenceBlock)
    (lengths : Nat → Int) : Prop :=
  match m,e,l,o,q with
  | .tensor m,.tensor e,.tensor l,.tensor o,.integer q =>
    0 < q ∧ InputContract ⟨⟨m,e,l,o,q.toNat,Bool.true⟩,rfl⟩ lengths
  | _,_,_,_,_ => False

theorem check_arguments_iff (m e l o : Checked.Argument) (q : Checked.SequenceBlock)
    (lengths : Nat → Int) :
    checkArguments m e l o q lengths=.ok () ↔ AdmittedArguments m e l o q lengths := by
  cases m <;> cases e <;> cases l <;> cases o <;> cases q <;>
    simp only [checkArguments,AdmittedArguments] <;> try simp
  split <;> simp_all [check_input_iff]

theorem arange_bounds (c : Call) (lengths : Nat → Int) (h : InputContract c lengths)
    (live : Checked.active c.val) :
    0 < Checked.block c.val ∧ Checked.block c.val ≤ 2^20 ∧ Checked.block c.val < 2^31 := by
  have valid := h.metadata.2.2.2.2.2.2.2.1
  have positive : 0 < Checked.width c.val := valid.1
  have covers : Checked.width c.val ≤ Checked.block c.val := valid.2.1
  have bound := h.arange live
  omega

#axiomsClean exact_domain
#axiomsClean check_input_iff
#axiomsClean empty_domain_unchanged
#axiomsClean check_arguments_iff
#axiomsClean arange_bounds
end VeriTile.Bench.Repairs.FlashDecode2PhiBlockChecked

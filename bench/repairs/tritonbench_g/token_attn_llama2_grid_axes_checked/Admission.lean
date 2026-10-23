import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Contract

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open Whole
noncomputable section

/-- The preceding Python guard's successful domain, stated on the same actual
Tensor observations and pointer-decoded metadata as the candidate contract. -/
def precedingAdmission (t : TensorCall) (words : Nat → Word) : Prop :=
  Header t ∧
    (t.call.dim=16 ∨ t.call.dim=32 ∨ t.call.dim=64 ∨ t.call.dim=128) ∧
    (0<t.call.kvHeads ∧ 0<t.call.heads ∧ t.call.heads%t.call.kvHeads=0) ∧
    envelope t.call<2147483648 ∧ MetadataChecks t.call (metadata t words) ∧
    (active t → DenseChecks t ∧ PhysicalChecks t)

/-- Exact successful admission correspondence: every preceding check is
retained, and the only new conjunction is the two live-call axis bounds. -/
theorem exact_candidate_admission (t : TensorCall) (words : Nat → Word) :
    InputContract t words ↔ precedingAdmission t words ∧ (active t → AxisChecks t) := by
  constructor
  · rintro ⟨header,guard,contents,live⟩
    refine ⟨⟨header,guard.dimension,?_,guard.signed,contents,
      fun h => ⟨(live h).1,(live h).2.1⟩⟩,fun h => (live h).2.2⟩
    exact (head_checks_iff _ _).mpr ⟨guard.kvPositive,guard.groupsPositive,guard.groupsExact⟩
  · rintro ⟨⟨header,dim,heads,bound,contents,live⟩,axes⟩
    have hg := (head_checks_iff _ _).mp heads
    exact ⟨header,⟨dim,hg.1,hg.2.1,hg.2.2,bound⟩,contents,
      fun h => ⟨(live h).1,(live h).2,axes h⟩⟩

theorem python_candidate_checks (t : TensorCall) (words : Nat → Word) :
    (∃ launch, checkInput t words=.ok launch) ↔
      precedingAdmission t words ∧ (active t → AxisChecks t) :=
  (checks_iff t words).trans (exact_candidate_admission t words)

theorem empty_preceding_admission (t : TensorCall) (words : Nat → Word) (empty : ¬active t) :
    InputContract t words ↔ precedingAdmission t words := by
  rw [exact_candidate_admission]
  simp [empty]

#axiomsClean exact_candidate_admission
#axiomsClean python_candidate_checks
#axiomsClean empty_preceding_admission
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked

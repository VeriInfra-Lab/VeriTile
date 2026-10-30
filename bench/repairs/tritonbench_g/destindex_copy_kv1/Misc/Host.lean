import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Kernel

/-! Optional historical wrapper theorem; not a public kernel obligation. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv1
open VeriTile.Triton FP.Structural
noncomputable section
structure Result where
  returned : Unit
  outputObject : Nat
  outputMetadata : VeriTile.Torch.TensorMetadata
  memory : Memory

def runWrapper (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (object : Nat)
    (before : Memory) (other : State Word) : Except InputError Result := do
  checkInputs m w
  if (config m).count=0 ∨ (config m).heads=0 ∨ (config m).dims=0 then
    return ⟨(),object,m .output,before⟩
  return ⟨(),object,m .output,execute rest m w before other⟩

theorem wrapper_correct (rest : Algebra Word) (m : Inputs) (w : Nat → Word)
    (h : InputContract m w) (object : Nat) (before : Memory) (other : State Word) :
    ∃ out, runWrapper rest m w object before other=.ok out ∧ out.returned=() ∧
      out.outputObject=object ∧ out.outputMetadata=m .output ∧
      RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
        (fun x => before (addr (base m .output) x))
        (fun x => out.memory (addr (base m .output) x)) ∧
      (∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
        out.memory cell=before cell) := by
  have accepted := (checkInputs_iff m w).mpr h
  by_cases empty : (config m).count=0 ∨ (config m).heads=0 ∨ (config m).dims=0
  · refine ⟨⟨(),object,m .output,before⟩,by simp [runWrapper,accepted,empty],rfl,rfl,rfl,?_,?_⟩
    · constructor
      · intro p a b hp ha hb; rcases empty with e|e|e <;> omega
      · intro x _; rfl
    · intro x _; rfl
  · exact ⟨⟨(),object,m .output,execute rest m w before other⟩,
      by simp [runWrapper,accepted,empty],rfl,rfl,rfl,execute_correct rest m w h before other⟩

#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv1

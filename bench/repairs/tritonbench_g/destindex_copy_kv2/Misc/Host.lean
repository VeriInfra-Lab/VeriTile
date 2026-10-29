import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Kernel
import bench.repairs.tritonbench_g.destindex_copy_kv1.Misc.Host
/-! Optional historical wrapper theorem; not a public kernel obligation. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv2
open VeriTile.Triton FP.Structural
noncomputable section
open Classical
abbrev Result := DestindexCopyKv1.Result
def runWrapper (m : Inputs) (w : Nat → Word) (object : Nat) (before : Memory) : Except InputError Result := do
  checkInputs m w
  if Nonempty m then
    return ⟨(),object,m .output,Physical.launch m (List.range (config m).count) before⟩
  return ⟨(),object,m .output,before⟩

theorem wrapper_correct (rest : Algebra Word) (m : Inputs) (w : Nat → Word) (h : InputContract m w)
    (object : Nat) (before : Memory) (binding : TensorBinding m w before) (other : State Word) :
    ∃ out, runWrapper m w object before=.ok out ∧ out.returned=() ∧ out.outputObject=object ∧
      out.outputMetadata=m .output ∧
      RowCopy (config m) (destination w) (fun x => before (addr (base m .source) x))
        (fun x => before (addr (base m .output) x))
        (fun x => out.memory (addr (base m .output) x)) ∧
      (∀ cell, (¬∃ x, touched (config m) (destination w) x ∧ cell=addr (base m .output) x) →
        out.memory cell=before cell) := by
  have accepted := (checkInputs_iff m w).mpr h
  by_cases ne : Nonempty m
  · refine ⟨⟨(),object,m .output,Physical.launch m (List.range (config m).count) before⟩,
      by simp [runWrapper,accepted,ne],rfl,rfl,rfl,?_⟩
    rw [Physical.launch_eq m w h ne]
    exact DestindexCopyKv1.Physical.correct rest m w h.common before binding other _ (by simp)
  · refine ⟨⟨(),object,m .output,before⟩,by simp [runWrapper,accepted,ne],rfl,rfl,rfl,?_,fun _ _ => rfl⟩
    constructor
    · intro p a b hp ha hb
      exact False.elim (ne ⟨by omega,by omega,by omega⟩)
    · intro x _; rfl

#axiomsClean Mathematical.step_memory
#axiomsClean Mathematical.correct
#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.DestindexCopyKv2

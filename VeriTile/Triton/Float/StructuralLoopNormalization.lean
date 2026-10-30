import VeriTile.Triton.Float.StructuralSafety

/-! Counted natural loops and unit-stride ranges bind the same index and
follow the same actual body successors, including failures and access traces. -/
namespace VeriTile.Triton.FP.Structural
noncomputable section

theorem loop_eq_range {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop : Nat) (body : List ComputeStmt) (s : State α) :
    loop M idx start stop body s = range M idx start stop 1 body s := by
  rw [loop, range]
  simp only [one_ne_zero, ↓reduceIte]
  split
  next active =>
    cases ran : run M body (s.setReg idx .nat [] (fun _ => start)) with
    | none => rfl
    | some t =>
      exact loop_eq_range M idx (start+1) stop body t
  next => rfl
termination_by stop-start
decreasing_by omega

namespace Safety

theorem loopSafe_eq_rangeSafe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (start stop : Nat)
    (body : List ComputeStmt) (s : State α) :
    loopSafe M bounds idx start stop body s = rangeSafe M bounds idx start stop 1 body s := by
  rw [loopSafe_unfold, rangeSafe_unfold]
  simp only [one_ne_zero, ↓reduceIte]
  split
  next active =>
    congr 1
    cases ran : run M body (s.setReg idx .nat [] (fun _ => start)) with
    | none => rfl
    | some t =>
      exact loopSafe_eq_rangeSafe M bounds idx (start+1) stop body t
  next => rfl
termination_by stop-start
decreasing_by omega

end Safety
end
end VeriTile.Triton.FP.Structural

import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit
namespace VeriTile.Triton.FP.Structural
noncomputable section

/-- The counted constructor and a unit-stride range visit the same indices and
states. This is a semantic bridge; neither source constructor is replaced. -/
theorem loop_eq_range_one {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop : Nat) (body : List ComputeStmt) (s : State α) :
    loop M idx start stop body s = range M idx start stop 1 body s := by
  rw [loop, range]
  simp only [show (1 : Nat) ≠ 0 by decide, if_false]
  split
  · cases h : run M body (s.setReg idx .nat [] (fun _ => start)) with
    | none => rfl
    | some t => exact loop_eq_range_one M idx (start+1) stop body t
  · rfl
termination_by stop-start
decreasing_by omega

namespace Safety
theorem loopSafe_iff_rangeSafe_one {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (start stop : Nat)
    (body : List ComputeStmt) (s : State α) :
    loopSafe M bounds idx start stop body s ↔ rangeSafe M bounds idx start stop 1 body s := by
  rw [loopSafe_unfold, rangeSafe_unfold]
  simp only [show (1 : Nat) ≠ 0 by decide, if_false]
  split
  · cases h : run M body (s.setReg idx .nat [] (fun _ => start)) with
    | none => rfl
    | some t => exact and_congr_right fun _ =>
        loopSafe_iff_rangeSafe_one M bounds idx (start+1) stop body t
  · rfl
termination_by stop-start
decreasing_by omega
end Safety
#axiomsClean loop_eq_range_one
#axiomsClean Safety.loopSafe_iff_rangeSafe_one
end
end VeriTile.Triton.FP.Structural

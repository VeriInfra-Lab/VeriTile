import VeriTile.Triton.Float.StructuralSafety
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.Conv2d.UnitRanges
open VeriTile.Triton FP.Structural
noncomputable section

/-- The pasted single-argument range uses the counted-loop constructor.
Connect it to the existing unit-stride invariant without changing the source. -/
theorem loop_unit_range {α : Type} [Inhabited α] (M : Algebra α) (name : RegName)
    (body : List ComputeStmt) (start stop : Nat) (s : State α) :
    loop M name start stop body s = range M name start stop 1 body s := by
  rw [loop, range]
  simp only [Nat.one_ne_zero, if_false]
  by_cases h : start < stop
  · simp only [h, if_true]
    cases hr : run M body (s.setReg name .nat [] (fun _ => start)) with
    | none => rfl
    | some t =>
      simpa only [Option.bind_some] using loop_unit_range M name body (start+1) stop t
  · simp only [h, if_false]
termination_by stop-start
decreasing_by omega

theorem loop_safe_unit_range {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (name : RegName) (body : List ComputeStmt)
    (start stop : Nat) (s : State α) :
    Safety.loopSafe M bounds name start stop body s ↔
      Safety.rangeSafe M bounds name start stop 1 body s := by
  rw [Safety.loopSafe, Safety.rangeSafe]
  simp only [Nat.one_ne_zero, if_false]
  by_cases h : start < stop
  · simp only [h, if_true]
    apply and_congr_right
    intro _
    cases hr : run M body (s.setReg name .nat [] (fun _ => start)) with
    | none => rfl
    | some t => exact loop_safe_unit_range M bounds name body (start+1) stop t
  · simp only [h, if_false]
termination_by stop-start
decreasing_by omega

#axiomsClean loop_unit_range
#axiomsClean loop_safe_unit_range
end
end VeriTile.Bench.Repairs.Conv2d.UnitRanges

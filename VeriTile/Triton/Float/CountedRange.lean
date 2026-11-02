import VeriTile.Triton.Float.StructuralSafety

/-! Exact execution and access-safety connection for Python range(stop) and
range(start,stop,1). It changes no loop body, bound, successor or observation. -/
namespace VeriTile.Triton.FP.Structural
noncomputable section

theorem loop_eq_range_one {α : Type} [Inhabited α] (M : Algebra α)
    (idx : RegName) (start stop : Nat) (body : List ComputeStmt) (s : State α) :
    loop M idx start stop body s=range M idx start stop 1 body s := by
  rw [loop,range]
  simp only [show (1:Nat)≠0 by decide,↓reduceIte]
  split
  · congr 1
    funext t
    exact loop_eq_range_one M idx (start+1) stop body t
  · rfl
termination_by stop-start
decreasing_by omega

theorem loopSafe_eq_rangeSafe_one {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (start stop : Nat)
    (body : List ComputeStmt) (s : State α) :
    Safety.loopSafe M bounds idx start stop body s ↔
      Safety.rangeSafe M bounds idx start stop 1 body s := by
  rw [Safety.loopSafe,Safety.rangeSafe]
  simp only [show (1:Nat)≠0 by decide,↓reduceIte]
  split
  · apply and_congr_right
    intro _
    cases run M body (s.setReg idx .nat [] (fun _=>start)) with
    | none=>rfl
    | some t=>exact loopSafe_eq_rangeSafe_one M bounds idx (start+1) stop body t
  · rfl
termination_by stop-start
decreasing_by omega
end
end VeriTile.Triton.FP.Structural

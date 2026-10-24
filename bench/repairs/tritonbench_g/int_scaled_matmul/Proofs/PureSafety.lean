import VeriTile.Triton.Float.ProgramIO

namespace VeriTile.Bench.Repairs.IntScaledMatmul.PureSafety
open VeriTile.Triton FP.Structural
noncomputable section

/-- A straight-line source fragment whose individual statements perform no
memory accesses needs no symbolic expansion of its computed register values. -/
theorem trace {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (pure : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st code ih =>
    apply Safety.traceSafeList_cons_intro
    · exact pure st (by simp) s
    · intro t _
      exact ih (fun q hq => pure q (by simp [hq])) t
end
end VeriTile.Bench.Repairs.IntScaledMatmul.PureSafety

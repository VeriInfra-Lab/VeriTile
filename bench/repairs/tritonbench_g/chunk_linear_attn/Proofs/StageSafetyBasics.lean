import bench.repairs.tritonbench_g.chunk_linear_attn.Grid
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
noncomputable section

/-- Reduce a final statement without evaluating its irrelevant successor. -/
theorem singleton {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (stmt : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [stmt] s ↔ Safety.stepSafe M bounds stmt s := by
  rw [Safety.traceSafeList_cons]
  simp

/-- A segment whose individual statements access no memory needs no value
execution expansion when establishing its access safety. -/
theorem pure_trace {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ s, Safety.stepSafe M bounds st s) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
      apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
      intro t _
      exact ih (fun statement mem => safe statement (by simp [mem])) t

theorem matrix_bound (heads rows cols head row col : Nat)
    (hh : head < heads) (hr : row < rows) (hc : col < cols) :
    head * (rows * cols) + row * cols + col < heads * rows * cols := by
  convert packed_lt (packed_lt hh hr) hc using 1 <;> ring

theorem history_bound (heads chunks rows cols head chunk row col : Nat)
    (hh : head < heads) (ht : chunk < chunks) (hr : row < rows) (hc : col < cols) :
    head * (chunks * rows * cols) + chunk * rows * cols + row * cols + col <
      heads * chunks * rows * cols := by
  convert packed_lt (packed_lt (packed_lt hh ht) hr) hc using 1 <;> ring

/-- Exactly the regions accessed by a forward/reverse state source.
Optional endpoint capacities are required only when their source branch runs. -/
structure StateCapacity (p : Parameters) (rev seed final : Bool)
    (bounds : RegionName → Nat) : Prop where
  left : p.B * p.H * p.T * p.K ≤ bounds (if rev then "query" else "key")
  right : p.B * p.H * p.T * p.V ≤ bounds (if rev then "incoming" else "value")
  history : p.B * p.H * p.NT * p.K * p.V ≤ bounds (historyName rev)
  initial : seed = Bool.true → p.B * p.H * p.K * p.V ≤ bounds (seedName rev)
  final : final = Bool.true → p.B * p.H * p.K * p.V ≤ bounds (finalName rev)

structure OutputCapacity (p : Parameters) (bounds : RegionName → Nat) : Prop where
  query : p.B * p.H * p.T * p.K ≤ bounds "query"
  key : p.B * p.H * p.T * p.K ≤ bounds "key"
  value : p.B * p.H * p.T * p.V ≤ bounds "value"
  history : p.B * p.H * p.NT * p.K * p.V ≤ bounds "history"
  output : p.B * p.H * p.T * p.V ≤ bounds "output"

structure GradientCapacity (p : Parameters) (bounds : RegionName → Nat) : Prop where
  query : p.B * p.H * p.T * p.K ≤ bounds "query"
  key : p.B * p.H * p.T * p.K ≤ bounds "key"
  value : p.B * p.H * p.T * p.V ≤ bounds "value"
  incoming : p.B * p.H * p.T * p.V ≤ bounds "incoming"
  history : p.B * p.H * p.NT * p.K * p.V ≤ bounds "history"
  historyGradient : p.B * p.H * p.NT * p.K * p.V ≤ bounds "historyGradient"
  queryGradient : p.B * p.H * p.T * p.K ≤ bounds "queryGradient"
  keyGradient : p.B * p.H * p.T * p.K ≤ bounds "keyGradient"
  valuePartials : keyBlocks p * (p.B * p.H) * p.T * p.V ≤ bounds "valuePartials"

end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety

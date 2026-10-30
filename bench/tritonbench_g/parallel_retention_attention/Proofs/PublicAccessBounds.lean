import VeriTile.Triton.Float.StructuralSafetyLoops
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicAccess
open VeriTile.Triton FP.Structural
noncomputable section

/-- Backing for both checked matrix axes, independent of tile offsets. -/
def MatrixBound (bounds : RegionName → Nat) (region : RegionName)
    (base rows cols rowStride colStride : Nat) : Prop :=
  ∀ i j, i < rows → j < cols →
    base + i * rowStride + j * colStride < bounds region

/-- Pure source segments do not require expansion of their arithmetic values.
The conclusion is the shared safety judgment of the actual evaluator. -/
theorem traceSafeList_of_steps {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (code : List ComputeStmt)
    (safe : ∀ st ∈ code, ∀ t, Safety.stepSafe M bounds st t) (s : State α) :
    Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => simp
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro
    · exact safe st (by simp) s
    · intro t ht
      exact ih (fun st h u => safe st (by simp [h]) u) t

/-- Normalize the concrete two-coordinate list before recursive safety
simplification. These are the shared evaluator predicates, not new effects. -/
theorem safeBlockPtr2 {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (bounds : RegionName → Nat) (s : State α)
    (region : RegionName) (base : Op .nat []) (parent block strides : List Nat)
    (a b : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets region base parent block strides [a,b]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s a ∧
        Safety.safeOp M precision bounds s b := by
  rw [Safety.safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

theorem safeAdvance2 {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (bounds : RegionName → Nat) (s : State α)
    (shape : TileShape) (pointer : Op .blockPtr shape) (a b : Op .int []) :
    Safety.safeOp M precision bounds s (.advanceBlockPtrDyn pointer [a,b]) ↔
      Safety.safeOp M precision bounds s pointer ∧ Safety.safeOp M precision bounds s a ∧
        Safety.safeOp M precision bounds s b := by
  rw [Safety.safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

/-- An assignment cannot alter another live pointer register. The evaluated
arithmetic expression is kept opaque throughout this source-state fact. -/
theorem assign_keeps_reg {α : Type} [Inhabited α] (M : Algebra α)
    (dtype shape name) (expr : ComputeExpr dtype shape) (s t : State α)
    (other : RegName) (different : other ≠ name) (d : TileDType) (sh : TileShape)
    (h : step M (.assign dtype shape name expr) s = some t) :
    t.regs d sh other=s.regs d sh other := by
  simp only [step] at h
  cases he : evalExpr M expr s with
  | none => simp [he] at h
  | some value =>
    simp only [he,bind,pure,Option.bind_some,Option.some.injEq] at h
    subst t
    simp [State.setReg,different]

#axiomsClean traceSafeList_of_steps
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicAccess

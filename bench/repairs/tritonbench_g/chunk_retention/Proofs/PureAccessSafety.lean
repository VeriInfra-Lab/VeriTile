import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.AccessSafety
open VeriTile Triton FP.Structural
noncomputable section

theorem singleton_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName→Nat) (statement : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [statement] s ↔ Safety.stepSafe M bounds statement s := by
  rw [Safety.traceSafeList_cons]
  simp only [Safety.traceSafeList_nil,implies_true,and_true]

/-- Expose concrete coordinate children before recursive safety rewriting.
Expanding safeOp under an unknown list member would unfold every Op case. -/
theorem two_offset_pointer {α : Type} [Inhabited α] (M : Algebra α)
    (precision : Option ComputeDType) (bounds : RegionName→Nat) (s : State α)
    (r : RegionName) (base : Op .nat []) (parent block strides : List Nat) (x y : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets r base parent block strides [x,y]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s x ∧
        Safety.safeOp M precision bounds s y := by
  rw [Safety.safeOp_unfold]
  simp [List.mem_cons]

/-- A separate nonterminal rule prevents simplification from evaluating a
terminal scatter whose successor is irrelevant to trace safety. -/
theorem pair_safe {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName→Nat) (first second : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds (first::second::rest) s ↔
      Safety.stepSafe M bounds first s ∧
      ∀ t,step M first s=some t → Safety.traceSafeList M bounds (second::rest) t :=
  Safety.traceSafeList_cons M bounds first (second::rest) s

/-- Memory-free source fragments need no evaluation of numerical intermediates
to establish access safety. Each actual statement must still be proved safe in
every entry state; this is a derived theorem, never an assumed IO callback. -/
theorem list_safe_of_steps {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName→Nat) (body : List ComputeStmt)
    (h : ∀ statement∈body,∀ s,Safety.stepSafe M bounds statement s) (s : State α) :
    Safety.traceSafeList M bounds body s := by
  induction body generalizing s with
  | nil => exact Safety.traceSafeList_nil M bounds s
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro
    · exact h st (List.mem_cons_self ..) s
    · intro next _
      exact ih (fun statement hm=>h statement (List.mem_cons_of_mem _ hm)) next

end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.AccessSafety

import VeriTile.Triton.Float.StructuralSafety

/-! Invariant introduction for the existing Structural loop safety judgments.

The invariant concerns the state immediately before binding the current loop
index. Each step proves safety of the actual body at that bound state and
preserves the invariant after the actual successful `run`. These lemmas do not
provide successful execution: public IO still proves that separately. No custom
executor or safety predicate is installed in the source semantics.
-/
namespace VeriTile.Triton.FP.Structural.Safety
noncomputable section

/-- A natural counted-loop invariant follows the actual body successor. -/
theorem loopSafe_of_invariant {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (stop : Nat)
    (body : List ComputeStmt) (P : Nat → State α → Prop)
    (bodySafe : ∀ q, q < stop → ∀ t, P q t →
      traceSafeList M bounds body (t.setReg idx .nat [] (fun _ => q)))
    (preserved : ∀ q, q < stop → ∀ t, P q t → ∀ u,
      run M body (t.setReg idx .nat [] (fun _ => q)) = some u → P (q + 1) u)
    (start : Nat) (s : State α) (initial : P start s) :
    loopSafe M bounds idx start stop body s := by
  rw [loopSafe_unfold]
  split
  next active =>
    refine ⟨bodySafe start active s initial, ?_⟩
    cases ran : run M body (s.setReg idx .nat [] (fun _ => start)) with
    | none => trivial
    | some t =>
      exact loopSafe_of_invariant M bounds idx stop body P bodySafe preserved
        (start + 1) t (preserved start active s initial t ran)
  next => trivial
termination_by stop - start
decreasing_by omega

/-- A natural strided-loop invariant. Zero stride and empty intervals retain
the evaluator's no-iteration behavior without a body obligation. -/
theorem rangeSafe_of_invariant {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (stop stride : Nat)
    (body : List ComputeStmt) (P : Nat → State α → Prop)
    (bodySafe : ∀ q, q < stop → ∀ t, P q t →
      traceSafeList M bounds body (t.setReg idx .nat [] (fun _ => q)))
    (preserved : ∀ q, q < stop → ∀ t, P q t → ∀ u,
      run M body (t.setReg idx .nat [] (fun _ => q)) = some u → P (q + stride) u)
    (start : Nat) (s : State α) (initial : P start s) :
    rangeSafe M bounds idx start stop stride body s := by
  rw [rangeSafe_unfold]
  split
  next => trivial
  next nonzero =>
    split
    next active =>
      refine ⟨bodySafe start active s initial, ?_⟩
      cases ran : run M body (s.setReg idx .nat [] (fun _ => start)) with
      | none => trivial
      | some t =>
        exact rangeSafe_of_invariant M bounds idx stop stride body P bodySafe preserved
          (start + stride) t (preserved start active s initial t ran)
    next => trivial
termination_by stop - start
decreasing_by omega

/-- Signed loop positions retain the signed index register, positive natural
stride and exact predecessor/successor states of `rangeIntSafe`. -/
theorem rangeIntSafe_of_invariant {α : Type} [Inhabited α] (M : Algebra α)
    (bounds : RegionName → Nat) (idx : RegName) (stop : Int) (stride : Nat)
    (body : List ComputeStmt) (P : Int → State α → Prop)
    (bodySafe : ∀ q, q < stop → ∀ t, P q t →
      traceSafeList M bounds body (t.setReg idx .int [] (fun _ => q)))
    (preserved : ∀ q, q < stop → ∀ t, P q t → ∀ u,
      run M body (t.setReg idx .int [] (fun _ => q)) = some u → P (q + stride) u)
    (start : Int) (s : State α) (initial : P start s) :
    rangeIntSafe M bounds idx start stop stride body s := by
  rw [rangeIntSafe_unfold]
  split
  next => trivial
  next nonzero =>
    split
    next active =>
      refine ⟨bodySafe start active s initial, ?_⟩
      cases ran : run M body (s.setReg idx .int [] (fun _ => start)) with
      | none => trivial
      | some t =>
        exact rangeIntSafe_of_invariant M bounds idx stop stride body P bodySafe preserved
          (start + stride) t (preserved start active s initial t ran)
    next => trivial
termination_by (stop - start).toNat
decreasing_by omega

end
end VeriTile.Triton.FP.Structural.Safety

import VeriTile.Triton.Float.IntegerStage
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety for the unchanged integer-stage evaluator.
Its custom llrint/finite/fma expression children are checked recursively.
Addresses and masks use the existing Structural evaluator. The trace follows
IntegerStage.step, including narrowed integer stores and their actual effects
on later addresses. Fallback control flow uses precisely the Structural
interpreter and safety: this does not extend llrint inside control flow.
Successful execution remains a separate IO obligation.
-/
namespace QuantizeIntegerStage
open VeriTile.Triton FP.Structural
namespace Safety
noncomputable section

def safeExpr {α : Type} [Inhabited α] (M : Algebra α)
    (bindings : RuntimeRegisters) (bounds : RegionName → Nat) (s : State α) :
    ComputeExpr dtype shape → Prop
  | .llrint arg | .isFinite _ arg => safeExpr M bindings bounds s arg
  | .fma _ _ _ a b c =>
      safeExpr M bindings bounds s a ∧ safeExpr M bindings bounds s b ∧ safeExpr M bindings bounds s c
  | .fmaTyped _ _ _ a b c =>
      safeExpr M bindings bounds s a ∧ safeExpr M bindings bounds s b ∧ safeExpr M bindings bounds s c
  | .alg e => FP.Structural.Safety.safeOp M none bounds s e
  | .compute e => FP.Structural.Safety.safeCompute M bounds s e
  | .opaque (dtype := .int) marker => (bindings marker).isSome = Bool.true
  | .opaque _ => False

/-- Custom assignments and stores use the custom expression checker.
All other effects have exactly the existing Structural interpretation. -/
def stepSafe {α : Type} [Inhabited α] (M : Algebra α)
    (bindings : RuntimeRegisters) (bounds : RegionName → Nat) :
    ComputeStmt → State α → Prop
  | .assign _ _ _ e, s => safeExpr M bindings bounds s e
  | .store _ _ mem value mask, s =>
      FP.Structural.Safety.safeAddress M none bounds s mem ∧ safeExpr M bindings bounds s value ∧
      FP.Structural.Safety.safeMask M none bounds s mask ∧
      FP.Structural.Safety.addressSafe M none bounds s mem (FP.Structural.Safety.maskActive M none s mask)
  | stmt, s => FP.Structural.Safety.stepSafe M bounds stmt s

def traceSafeList {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (storage : IntegerStorage) (bindings : RuntimeRegisters)
    (bounds : RegionName → Nat) : List ComputeStmt → State α → Prop
  | [], _ => True
  | stmt :: rest, s => stepSafe M bindings bounds stmt s ∧
      match QuantizeIntegerStage.step M round storage bindings stmt s with
      | some t => traceSafeList M round storage bindings bounds rest t
      | none => True

abbrev safeExpr_unfold := @safeExpr.eq_def
abbrev stepSafe_unfold := @stepSafe.eq_def
abbrev traceSafeList_unfold := @traceSafeList.eq_def

@[simp] theorem traceSafeList_nil {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (storage : IntegerStorage) (bindings : RuntimeRegisters)
    (bounds : RegionName → Nat) (s : State α) :
    traceSafeList M round storage bindings bounds [] s := trivial

theorem traceSafeList_cons {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (storage : IntegerStorage) (bindings : RuntimeRegisters)
    (bounds : RegionName → Nat) (stmt : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    traceSafeList M round storage bindings bounds (stmt :: rest) s ↔
      stepSafe M bindings bounds stmt s ∧
        ∀ t, QuantizeIntegerStage.step M round storage bindings stmt s = some t →
          traceSafeList M round storage bindings bounds rest t := by
  simp only [traceSafeList]
  cases h : QuantizeIntegerStage.step M round storage bindings stmt s <;> simp

theorem traceSafeList_cons_of_step {α : Type} [Inhabited α] {M : Algebra α}
    {round : Option (Llrint64 α)} {storage : IntegerStorage} {bindings : RuntimeRegisters}
    {bounds : RegionName → Nat} {stmt : ComputeStmt} {rest : List ComputeStmt} {s t : State α}
    (head : stepSafe M bindings bounds stmt s)
    (executed : QuantizeIntegerStage.step M round storage bindings stmt s = some t)
    (tail : traceSafeList M round storage bindings bounds rest t) :
    traceSafeList M round storage bindings bounds (stmt :: rest) s := by
  rw [traceSafeList, executed]
  exact ⟨head, tail⟩

theorem traceSafeList_append {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (storage : IntegerStorage) (bindings : RuntimeRegisters)
    (bounds : RegionName → Nat) (before after : List ComputeStmt) (s : State α) :
    traceSafeList M round storage bindings bounds (before ++ after) s ↔
      traceSafeList M round storage bindings bounds before s ∧
      ∀ t, QuantizeIntegerStage.run M round storage bindings before s = some t →
        traceSafeList M round storage bindings bounds after t := by
  induction before generalizing s with
  | nil => simp [QuantizeIntegerStage.run]
  | cons stmt rest ih =>
    cases h : QuantizeIntegerStage.step M round storage bindings stmt s <;>
      simp [traceSafeList, QuantizeIntegerStage.run, h, ih, and_assoc]
end
end Safety

def TraceSafe {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (bounds : RegionName → Nat)
    (kernel : TypedKernel) (s : State α) : Prop :=
  Safety.traceSafeList M round kernel.integerStorage kernel.runtimeRegisters bounds kernel.program.surfaceBody s

theorem run_append {α : Type} [Inhabited α] (M : Algebra α)
    (round : Option (Llrint64 α)) (storage : IntegerStorage) (bindings : RuntimeRegisters)
    (before after : List ComputeStmt) (s : State α) :
    run M round storage bindings (before ++ after) s =
      (run M round storage bindings before s).bind (run M round storage bindings after) := by
  induction before generalizing s with
  | nil => rfl
  | cons stmt rest ih =>
    cases h : QuantizeIntegerStage.step M round storage bindings stmt s <;>
      simp [run, h, ih]
end QuantizeIntegerStage

import VeriTile.Triton.Float.ClosedStructuralAudit

namespace VeriTile.Tests.ClosedStructuralAudit
open scoped VeriTile.Spec

-- A closed syntax proof has neither numerical atoms nor strict certificates.
theorem identity (code : List Nat) : code ≡[[]] code :=
  VeriTile.Spec.FloatingPoint.refl code
/-- info: FP assumptions used by identity:
  none -/
#guard_msgs in
#print_fp_assumptions identity closed_structural

-- A public derivation premise must remain visible, including through a record.
theorem forwarded (code : List Nat) (h : code ≡[[]] code) : code ≡[[]] code := h
/-- error: closed structural audit rejects external FP proof/model premises -/
#guard_msgs in
#print_fp_assumptions forwarded closed_structural

structure Wrapped where
  certificate : ([] : List Nat) ≡[[]] []
theorem wrapped (w : Wrapped) : ([] : List Nat) ≡[[]] [] := w.certificate
/-- error: closed structural audit rejects external FP proof/model premises -/
#guard_msgs in
#print_fp_assumptions wrapped closed_structural

-- The outer theorem being transparent does not make an opaque dependency visible.
opaque hidden : ([] : List Nat) ≡[[]] [] := identity []
theorem opaqueUse : ([] : List Nat) ≡[[]] [] := hidden
/-- error: closed structural audit cannot inspect opaque project dependency VeriTile.Tests.ClosedStructuralAudit.hidden -/
#guard_msgs in
#print_fp_assumptions opaqueUse closed_structural

namespace Strict
abbrev model : VeriTile.Spec.StatementTheory Nat := ⟨"unit-test", fun _ _ => True⟩
instance view : VeriTile.Spec.ProgramSyntax Unit where
  Statement := Nat
  theory := model
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
theorem numerical : () ≡[[]] () :=
  VeriTile.Spec.FloatingPoint.ofDerivation rfl trivial
    (@VeriTile.Spec.Derivation.strict Nat model [] [] [] trivial)
/-- error: closed structural audit found numerical evidence constructor VeriTile.Spec.Derivation.strict -/
#guard_msgs in
#print_fp_assumptions numerical closed_structural
end Strict

namespace Numerical
instance view : VeriTile.Spec.ProgramSyntax Bool where
  Statement := Nat
  Signature := Unit
  signature := fun _ => ()
  body := fun _ => []
  structural := some (fun _ _ => True)
  numerical := fun _ _ _ => True
theorem numerical : false ≡[[]] true :=
  VeriTile.Spec.FloatingPoint.ofNumerical rfl rfl trivial
/-- error: closed structural audit found numerical evidence constructor VeriTile.Spec.ProgramDerivation.numerical -/
#guard_msgs in
#print_fp_assumptions numerical closed_structural
end Numerical

#axiomsClean identity
#axiomsClean Strict.numerical
#axiomsClean Numerical.numerical
end VeriTile.Tests.ClosedStructuralAudit

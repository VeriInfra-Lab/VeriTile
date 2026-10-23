import VeriTile.Triton.Float.ScalarOpsLaws
import VeriTile.Triton.Float.CountConversionExact
import VeriTile.Triton.Float.Counterexamples
import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.CountConversion
import VeriTile.Meta.StatementAudit

/-! Boundaries of strict scalar rules, checked by Lean rather than sampling.
These regressions distinguish word equality from numerical equality, preserve
flush/cast behavior, and prevent exact count facts from becoming unbounded.
The existing counterexamples supply the reassociation and FMA witnesses. -/
namespace FPStrictRuleBoundaries
open VeriTile.Triton.FP
open VeriTile

set_option maxRecDepth 4096

/-- No finiteness, sign, format or configuration premise may be added to the
shared all-word commutation rule. In particular this includes NaNs and ±0. -/
theorem all_words_add_comm (f : Format) (cfg : Config) (a b : Value f) :
    add cfg a b = add cfg b a :=
  ScalarOpsLaws.add_comm cfg a b

theorem all_words_mul_comm (f : Format) (cfg : Config) (a b : Value f) :
    mul cfg a b = mul cfg b a :=
  ScalarOpsLaws.mul_comm cfg a b

/-- Finite -0 is already a counterexample to unconditional bitwise ADD-ZERO.
The numerical projection alone would hide this difference. -/
theorem add_zero_changes_negative_zero :
    add .ieeeValue (Value.zero .fp32 true) (Value.zero .fp32) = Value.zero .fp32 ∧
    add .ieeeValue (Value.zero .fp32 true) (Value.zero .fp32) ≠ Value.zero .fp32 true ∧
    (Value.zero .fp32 true).toRat? = (Value.zero .fp32).toRat? := by
  decide +kernel

/-- Commutation does not erase the configuration's zero-sign choice. -/
theorem directed_rounding_keeps_negative_zero :
    add ⟨.towardNegative, false, false⟩ (Value.zero .fp32 true) (Value.zero .fp32) =
      Value.zero .fp32 true := by
  decide +kernel

/-- Distinct NaN payloads do not invalidate the all-word theorem: this model
canonicalizes them in either order, without assuming finite inputs. -/
theorem exceptional_inputs_commute :
    add .ieeeValue (Value.ofBits .fp32 0x7f800001) (Value.ofBits .fp32 0xffc00001) =
      Value.qNaN .fp32 ∧
    mul .ieeeValue (Value.zero .fp32 true) (Value.infinity .fp32) = Value.qNaN .fp32 := by
  decide +kernel

/-- The least positive subnormal survives ×1 without flushing. Either flush
switch changes its bits, so a theorem for all configurations must retain cast. -/
theorem multiply_one_flush_boundary :
    let x := Value.ofBits .fp32 1
    let one := ScalarOpsLaws.one .fp32
    mul .ieeeValue x one = x ∧
    mul ⟨.nearestEven, true, false⟩ x one = Value.zero .fp32 ∧
    mul ⟨.nearestEven, false, true⟩ x one = Value.zero .fp32 ∧
    mul ⟨.nearestEven, true, false⟩ x one ≠ x ∧
    mul ⟨.nearestEven, false, true⟩ x one ≠ x := by
  decide +kernel

/-- The safe all-word result keeps normalization explicit instead of silently
strengthening multiplication/division by one to identity on input bits. -/
theorem one_operations_keep_cast (f : Format) (cfg : Config) (x : Value f) :
    mul cfg x (ScalarOpsLaws.one f) = cast f cfg x ∧
    div cfg x (ScalarOpsLaws.one f) = cast f cfg x :=
  ⟨ScalarOpsLaws.mul_one_cast cfg x, ScalarOpsLaws.div_one_cast cfg x⟩

theorem count_last_exact_value :
    (CountConversion.Exact.fromNat (2^24)).decode = .finite false (2^24 : Rat) := by
  exact CountConversion.Exact.fromNat_decode (2^24) (by decide)

/-- The next integer is not represented exactly. The successor equation still
happens to hold at i=2^24, but already fails at i=2^24+1; it is not unbounded. -/
theorem count_first_inexact_value :
    (CountConversion.Exact.fromNat (2^24+1)).toRat? = some (2^24 : Rat) ∧
    (CountConversion.Exact.fromNat (2^24+1)).toRat? ≠ some (2^24+1 : Rat) := by
  decide +kernel

theorem count_successor_cannot_be_unbounded :
    CountConversion.Exact.fromNat ((2^24+1)+1) ≠
      add .ieeeValue (CountConversion.Exact.fromNat (2^24+1))
        (CountConversion.Exact.fromNat 1) := by
  decide +kernel

/-- Reuse the published exact witness: strict commutation must not introduce
an unconditional association rule for the same concrete model. -/
theorem no_unconditional_add_association :
    ¬ ∀ a b c : Value .fp32,
      add .ieeeValue (add .ieeeValue a b) c = add .ieeeValue a (add .ieeeValue b c) := by
  intro h
  have equal := congrArg Value.toRat? (h (Value.ofBits .fp32 0x4b800000)
    (Value.ofBits .fp32 0x3f800000) (Value.ofBits .fp32 0xcb800000))
  have witness := fp32_add_assoc_witness
  rw [witness.1, witness.2] at equal
  norm_num at equal

theorem no_unconditional_fma_contraction :
    ¬ ∀ a b c : Value .fp32,
      add .ieeeValue (mul .ieeeValue a b) c = fma .ieeeValue a b c := by
  intro h
  have equal := congrArg Value.toRat? (h (Value.ofBits .fp32 0x3f800400)
    (Value.ofBits .fp32 0x3f7ff800) (Value.ofBits .fp32 0xbf800000))
  have witness := fp32_fma_witness
  rw [witness.1, witness.2] at equal
  norm_num [mkRat] at equal

/-- The fixed model's exact rules are usable with an empty empirical rule
list. These theorem signatures cannot request EvidenceValidated or Models. -/
theorem strict_commutation :
    Spec.Derivation ([] : Spec.Assumptions GuardedFragment)
      [ScalarArithmetic.lhs .addCommute] [ScalarArithmetic.rhs .addCommute] :=
  ScalarArithmetic.strict_admitted [] .addCommute (by decide)

theorem strict_multiplication_commutation :
    Spec.Derivation ([] : Spec.Assumptions GuardedFragment)
      [ScalarArithmetic.lhs .mulCommute] [ScalarArithmetic.rhs .mulCommute] :=
  ScalarArithmetic.strict_admitted [] .mulCommute (by decide)

theorem strict_count_successor :
    Spec.Equiv [CountConversion.lhs .successor] [CountConversion.rhs .successor]
      ([] : Spec.Assumptions GuardedFragment) :=
  CountConversion.strict_rewrite .successor

/-- ADD-ZERO retains empirical status because its original finite guard
includes -0; exact commutation and finite/no-flush ×1 do not need that status. -/
theorem rule_evidence_boundary :
    ScalarArithmetic.Atom.entry? .addCommute = none ∧
    ScalarArithmetic.Atom.entry? .mulCommute = none ∧
    ScalarArithmetic.Atom.entry? .mulOne = none ∧
    ScalarArithmetic.Atom.entry? .divOne = none ∧
    (ScalarArithmetic.Atom.entry? .addZero).isSome = true ∧
    (ScalarArithmetic.Atom.entry? .addAssociate).isSome = true := by
  decide +kernel

theorem admitted_association (R : ScalarArithmetic.Rules) :
    Spec.Derivation R.assumptions
      [ScalarArithmetic.lhs .addAssociate] [ScalarArithmetic.rhs .addAssociate] :=
  ScalarArithmetic.admitted R .addAssociate (by decide)

/-- info: FP assumptions used by strict_commutation:
---
info:   none
---
info: Strict model certificates used by strict_commutation:
---
info:   model: VeriTile.Software.strictFP32/ieeeValue/maxnum/documented-exp-special-values
    certificate: List.Forall₂.cons
  (ScalarArithmetic.strict_certificate ScalarArithmetic.Atom.addCommute (of_decide_eq_true (id (Eq.refl true))))
  List.Forall₂.nil -/
#guard_msgs in
#print_fp_assumptions strict_commutation

/-- info: FP assumptions used by strict_multiplication_commutation:
---
info:   none
---
info: Strict model certificates used by strict_multiplication_commutation:
---
info:   model: VeriTile.Software.strictFP32/ieeeValue/maxnum/documented-exp-special-values
    certificate: List.Forall₂.cons
  (ScalarArithmetic.strict_certificate ScalarArithmetic.Atom.mulCommute (of_decide_eq_true (id (Eq.refl true))))
  List.Forall₂.nil -/
#guard_msgs in
#print_fp_assumptions strict_multiplication_commutation

/-- info: FP assumptions used by strict_count_successor:
---
info:   none
---
info: Strict model certificates used by strict_count_successor:
---
info:   model: VeriTile.Software.strictFP32/ieeeValue/maxnum/documented-exp-special-values
    certificate: List.Forall₂.cons (CountConversion.strict_sound CountConversion.Atom.successor) List.Forall₂.nil -/
#guard_msgs in
#print_fp_assumptions strict_count_successor

-- Only the selected experimental branch may be reported. The impossible
-- strict branch inside the generic dispatcher is not evidence for associativity.
/-- info: FP assumptions used by admitted_association:
---
info:   add_assoc -/
#guard_msgs in
#print_fp_assumptions admitted_association

#axiomsClean all_words_add_comm
#axiomsClean all_words_mul_comm
#axiomsClean add_zero_changes_negative_zero
#axiomsClean directed_rounding_keeps_negative_zero
#axiomsClean exceptional_inputs_commute
#axiomsClean multiply_one_flush_boundary
#axiomsClean one_operations_keep_cast
#axiomsClean count_last_exact_value
#axiomsClean count_first_inexact_value
#axiomsClean count_successor_cannot_be_unbounded
#axiomsClean no_unconditional_add_association
#axiomsClean no_unconditional_fma_contraction
#axiomsClean strict_commutation
#axiomsClean strict_multiplication_commutation
#axiomsClean strict_count_successor
#axiomsClean rule_evidence_boundary
#axiomsClean admitted_association

end FPStrictRuleBoundaries

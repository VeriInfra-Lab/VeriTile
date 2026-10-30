import VeriTile.Meta.FPProve
import VeriTile.Meta.StatementAudit

/-! The strict certificate's theory is part of the derivation type. These
fixtures exercise tactic composition and evidence reporting independently of
any experimental or floating-point implementation details. -/
namespace FPStrictMetaTests
open VeriTile Spec

private abbrev natTheory : StatementTheory Nat where
  modelID := "exact-natural-addition-test"
  Certified := Eq

local instance : StatementTheory Nat := natTheory

private theorem commutationCertificate (a b : Nat) :
    natTheory.Certified [a + b] [b + a] :=
  congrArg (fun n => [n]) (Nat.add_comm a b)

@[spec_rule] private theorem commute {R : Assumptions Nat} (a b : Nat) :
    Derivation R [a + b] [b + a] :=
  .strict (commutationCertificate a b)

theorem strict_decomposition (a b : Nat) :
    Derivation [] [10, a + b, 20] [10, b + a, 20] := by
  equiv_decompose
  all_goals fp_prove

theorem strict_search (a b : Nat) :
    Derivation [] [10, a + b, 20] [10, b + a, 20] := by
  fp_prove

private def experimentalEntry : RuleEntry Nat where
  rule := {
    lhs := [7], rhs := [8]
    contract := {
      ruleID := "TEST-RELAXATION"
      instanceKey := "strict-meta-test"
      configuration := .null
      warningPolicy := .passOnly } }
  evidence := {
    instanceKey := "strict-meta-test"
    artifact := some "test-fixture"
    bias := .pass
    vars := .pass }

-- The synthetic atom is usable only with the explicit admission premise.
theorem mixed (a b : Nat) (accepted : AcceptedAtom experimentalEntry) :
    Derivation [experimentalEntry] [a + b, 7] [b + a, 8] := by
  equiv_decompose
  · fp_prove
  · exact .atom experimentalEntry (by simp) accepted

private theorem replay {R : Assumptions Nat} {lhs rhs : List Nat}
    (proof : Derivation R lhs rhs) : Derivation R lhs rhs := by
  induction proof with
  | refl code => exact .refl code
  | atom entry member accepted => exact .atom entry member accepted
  | symm proof ih => exact .symm ih
  | trans left right ihLeft ihRight => exact .trans ihLeft ihRight
  | frame before after proof ih => exact .frame before after ih
  | strict certificate => exact .strict certificate

theorem forwarded_strict (a b : Nat) : Derivation [] [a + b] [b + a] :=
  replay (commute a b)

theorem forwarded_mixed (a b : Nat) (accepted : AcceptedAtom experimentalEntry) :
    Derivation [experimentalEntry] [a + b, 7] [b + a, 8] :=
  replay (mixed a b accepted)

theorem conditional_certificate (a b : Nat)
    (certificate : natTheory.Certified [a] [b]) : Derivation [] [a] [b] :=
  .strict certificate

theorem unreachable_strict (accepted : AcceptedAtom experimentalEntry) :
    Derivation [experimentalEntry] [7] [8] := by
  by_cases impossible : Bool.false = Bool.true
  · exact .strict (False.elim (Bool.noConfusion impossible))
  · exact .atom experimentalEntry (by simp) accepted

set_option linter.unusedVariables false in
theorem unreachable_opaque (a b : Nat)
    (hidden : Derivation [] [a + b] [b + a]) :
    Derivation [] [a + b] [b + a] := by
  by_cases selected : Bool.true = Bool.true
  · exact commute a b
  · exact hidden

-- A certificate in another model is not silently accepted by the tactic.
set_option linter.unusedVariables false in
example (a b : Nat) : True := by
  fail_if_success
    have : @Derivation Nat (StatementTheory.empty Nat) [] [a + b] [b + a] := by
      fp_prove
  trivial

/-- info: FP assumptions used by strict_decomposition:
---
info:   none
---
info: Strict model certificates used by strict_decomposition:
---
info:   model: exact-natural-addition-test
    certificate: commutationCertificate a b -/
#guard_msgs in
#print_fp_assumptions strict_decomposition

/-- info: FP assumptions used by mixed:
---
info:   test_relaxation
---
info: Strict model certificates used by mixed:
---
info:   model: exact-natural-addition-test
    certificate: commutationCertificate a b -/
#guard_msgs in
#print_fp_assumptions mixed

/-- info: FP assumptions used by forwarded_strict:
---
info:   none
---
info: Strict model certificates used by forwarded_strict:
---
info:   model: exact-natural-addition-test
    certificate: commutationCertificate a b -/
#guard_msgs in
#print_fp_assumptions forwarded_strict

/-- info: FP assumptions used by forwarded_mixed:
---
info:   test_relaxation
---
info: Strict model certificates used by forwarded_mixed:
---
info:   model: exact-natural-addition-test
    certificate: commutationCertificate a b -/
#guard_msgs in
#print_fp_assumptions forwarded_mixed

/-- info: FP assumptions used by conditional_certificate:
---
info:   unresolved FP proof: certificate (atomic assumptions unavailable)
---
info: Strict model certificates used by conditional_certificate:
---
info:   model: exact-natural-addition-test
    certificate: certificate
    conditional on proof premises: certificate -/
#guard_msgs in
#print_fp_assumptions conditional_certificate

/-- info: FP assumptions used by unreachable_strict:
---
info:   test_relaxation -/
#guard_msgs in
#print_fp_assumptions unreachable_strict

/-- info: FP assumptions used by unreachable_opaque:
---
info:   none
---
info: Strict model certificates used by unreachable_opaque:
---
info:   model: exact-natural-addition-test
    certificate: commutationCertificate a b -/
#guard_msgs in
#print_fp_assumptions unreachable_opaque

#guard_msgs (drop info) in
#axiomsClean strict_decomposition
#guard_msgs (drop info) in
#axiomsClean strict_search
#guard_msgs (drop info) in
#axiomsClean mixed
#guard_msgs (drop info) in
#axiomsClean forwarded_mixed


/-- Abstract data must not make a structural derivation appear to use an
opaque numerical proof. The theory itself remains an explicit parameter. -/
theorem structural_data {α : Type} [StatementTheory α] (payload : α) :
    Derivation [] [payload] [payload] := .refl _

/-- A numerical premise applied to abstract data must still be reported. -/
theorem opaque_from_data {α : Type} (payload : α) (a b : Nat)
    (obtainProof : α → Derivation [] [a] [b]) : Derivation [] [a] [b] :=
  obtainProof payload

structure PackedProof (a b : Nat) where
  proof : Derivation [] [a] [b]

/-- Proof-bearing records are data, but their proof projections are inspected. -/
theorem opaque_projection (a b : Nat) (packed : PackedProof a b) :
    Derivation [] [a] [b] := packed.proof

theorem certificate_from_data {α : Type} (payload : α) (a b : Nat)
    (certificate : α → natTheory.Certified [a] [b]) : Derivation [] [a] [b] :=
  .strict (certificate payload)

/-- info: FP assumptions used by structural_data:
---
info:   none -/
#guard_msgs in
#print_fp_assumptions structural_data

/-- info: FP assumptions used by opaque_from_data:
---
info:   unresolved FP proof: obtainProof payload (atomic assumptions unavailable) -/
#guard_msgs in
#print_fp_assumptions opaque_from_data

/-- info: FP assumptions used by opaque_projection:
---
info:   unresolved FP proof: packed.1 (atomic assumptions unavailable) -/
#guard_msgs in
#print_fp_assumptions opaque_projection

/-- info: FP assumptions used by certificate_from_data:
---
info:   unresolved FP proof: certificate payload (atomic assumptions unavailable)
---
info: Strict model certificates used by certificate_from_data:
---
info:   model: exact-natural-addition-test
    certificate: certificate payload
    conditional on proof premises: certificate -/
#guard_msgs in
#print_fp_assumptions certificate_from_data

#guard_msgs (drop info) in
#axiomsClean structural_data
#guard_msgs (drop info) in
#axiomsClean opaque_from_data
#guard_msgs (drop info) in
#axiomsClean opaque_projection
#guard_msgs (drop info) in
#axiomsClean certificate_from_data

end FPStrictMetaTests

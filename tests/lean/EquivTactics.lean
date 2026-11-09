import VeriTile.Meta.FPProve
import VeriTile.Meta.StatementAudit
import bench.examples.FlatVectorAdd.FPEquiv
import bench.examples.VectorAdd.FPEquiv
import bench.examples.FloatDTypeAdd.FPEquiv
import bench.examples.HyperConnectionsWidth.FPEquiv

/-!
Proof-producing decomposition and search: multiple differences, insertion,
deletion, symmetry, transitivity, model metadata and missing assumptions.
The abstract Nat statements test structure without any floating-point model.
-/
namespace VeriTile.Tests.EquivTactics
set_option linter.unusedVariables false
open VeriTile Triton
open scoped VeriTile.Spec

theorem two_differences (R : Spec.Assumptions Nat)
    (ha : Spec.Derivation R [2] [3]) (hb : Spec.Derivation R [4] [5]) :
    [0, 2, 4, 9] ≡[R] [0, 3, 5, 9] := by
  split_equiv
  · change Spec.Derivation R [2] [3]
    exact ha
  · change Spec.Derivation R [4] [5]
    exact hb

theorem repeated_anchors (R : Spec.Assumptions Nat)
    (ha : Spec.Derivation R [2] [3]) (hb : Spec.Derivation R [4] [5]) :
    [0, 2, 0, 4, 0] ≡[R] [0, 3, 0, 5, 0] := by
  split_equiv
  all_goals solve_fp

theorem insertion (R : Spec.Assumptions Nat) (h : Spec.Derivation R [] [3, 4]) :
    [0, 9] ≡[R] [0, 3, 4, 9] := by
  split_equiv
  change Spec.Derivation R [] [3, 4]
  solve_fp

theorem deletion (R : Spec.Assumptions Nat) (h : Spec.Derivation R [] [3, 4]) :
    [0, 3, 4, 9] ≡[R] [0, 9] := by
  split_equiv
  solve_fp

theorem intact_block (R : Spec.Assumptions Nat) (h : Spec.Derivation R [2, 4] [3, 5]) :
    [0, 2, 4, 9] ≡[R] [0, 3, 5, 9] := by
  split_equiv (split := false)
  change Spec.Derivation R [2, 4] [3, 5]
  solve_fp

theorem empty_identity (R : Spec.Assumptions Nat) : ([] : List Nat) ≡[R] [] := by
  split_equiv

theorem symbolic_identity (R : Spec.Assumptions Nat) (xs : List Nat) : xs ≡[R] xs := by
  split_equiv

theorem symbolic_obligation (R : Spec.Assumptions Nat) (xs ys : List Nat)
    (h : Spec.Derivation R xs ys) : xs ≡[R] ys := by
  split_equiv
  change Spec.Derivation R xs ys
  exact h

theorem transitive_reverse (R : Spec.Assumptions Nat)
    (h1 : Spec.Derivation R [2] [3]) (h2 : Spec.Derivation R [3] [4]) :
    Spec.Derivation R [4] [2] := by
  fail_if_success solve_fp (maxSteps := 1)
  solve_fp (maxSteps := 2)

theorem contextual_search (R : Spec.Assumptions Nat)
    (h1 : Spec.Derivation R [2] [3]) (h2 : Spec.Derivation R [3] [4]) :
    Spec.Derivation R [8, 2, 9] [8, 4, 9] := by
  solve_fp

-- Explicit hints can discharge guards from local premises through another lemma.
private theorem guarded_step (R : Spec.Assumptions Nat) (P Q : Prop)
    (h : P → Spec.Derivation R [2] [3]) (back : Q → P) (q : Q) :
    Spec.Derivation R [2] [3] := h (back q)

theorem hinted_step (R : Spec.Assumptions Nat) (P Q : Prop)
    (h : P → Spec.Derivation R [2] [3]) (back : Q → P) (q : Q) :
    Spec.Derivation R [2] [3] := by
  solve_fp [guarded_step]

theorem domain_is_required (R : Spec.Assumptions Nat) (P : Prop)
    (h : P → Spec.Derivation R [2] [3]) : True := by
  fail_if_success
    have : Spec.Derivation R [2] [3] := by solve_fp
  trivial

-- A custom execution view ensures body equality never erases signature or
-- private-context obligations, including a structural-view ProgramDerivation.
structure Program where
  port : Nat
  scratch : Nat
  code : List Nat

instance : Spec.ProgramSyntax Program where
  Statement := Nat
  Signature := Nat
  signature := Program.port
  body := Program.code
  structural := some Eq
  sameContext := fun a b => a.scratch = b.scratch

theorem metadata_preserved (R : Spec.Assumptions Nat) (a b c d : Nat)
    (ports : a = b) (context : c = d) (h : Spec.Derivation R [2] [3]) :
    (⟨a, c, [0, 2, 9]⟩ : Program) ≡[R] ⟨b, d, [0, 3, 9]⟩ := by
  split_equiv
  · change a = b
    exact ports
  · change c = d
    exact context
  · change Spec.Derivation R [2] [3]
    solve_fp

theorem structural_view (R : Spec.Assumptions Nat) (h : Spec.Derivation R [2] [3]) :
    Spec.ProgramDerivation Eq R (⟨0, 0, [8, 2, 9]⟩ : Program) ⟨0, 0, [8, 3, 9]⟩ := by
  split_equiv
  solve_fp

-- Failed attempts must not solve the goal, silently accept missing admissions,
-- drop changed metadata, or discard insertions/deletions.
theorem rejected_changes : True := by
  fail_if_success
    have : ([1] : List Nat) ≡[([] : Spec.Assumptions Nat)] [2] := by
      split_equiv
      solve_fp
  fail_if_success
    have : ([] : List Nat) ≡[([] : Spec.Assumptions Nat)] [2] := by
      split_equiv
      solve_fp
  fail_if_success
    have : (⟨0, 0, [2]⟩ : Program) ≡[[]] ⟨1, 0, [2]⟩ := by
      split_equiv
      all_goals solve_fp
  fail_if_success
    have : (⟨0, 0, [2]⟩ : Program) ≡[[]] ⟨0, 1, [2]⟩ := by
      split_equiv
      all_goals solve_fp
  trivial

-- Merely having two-hop rewrites and cycles does not prove an unrelated goal.
theorem bounded_failure (R : Spec.Assumptions Nat)
    (h : Spec.Derivation R [2] [3]) : True := by
  fail_if_success
    have : Spec.Derivation R [2] [4] := by solve_fp
  trivial

open VeriTile.Bench.Examples

-- Imported atoms retain their guards when the syntax tactics compose them.
open FP.ScalarArithmetic in
theorem imported_fp_rule (R : Rules) : [lhs .addCommute] ≡[R] [rhs .addCommute] := by
  split_equiv
  solve_fp [admitted R .addCommute (by decide)]

open FP.ScalarArithmetic in
theorem two_fp_sites (R : Rules) :
    [lhs .mulCommute, lhs .mulCommute] ≡[R] [rhs .mulCommute, rhs .mulCommute] := by
  split_equiv
  all_goals solve_fp [admitted R .mulCommute (by decide)]

def add64 (B : Nat) : List ComputeStmt :=
  [.assign .real [B] "out" (.compute (.alg .fp64
    (.add .real (.consSame .nil) (.ref .real [B] "y") (.ref .real [B] "x"))))]

open FP.ScalarArithmetic in
theorem fp_precision_and_admission (R : Rules) : True := by
  fail_if_success
    have : Spec.Derivation R.assumptions [lhs .addCommute] [⟨guards .addCommute, add64 1, .real, none⟩] := by
      solve_fp [admitted R .addCommute (by decide)]
  fail_if_success
    have : Spec.Derivation ([] : Spec.Assumptions FP.GuardedFragment)
        [lhs .addCommute] [rhs .addCommute] := by
      solve_fp
  -- The guarded relation cannot be used as its unguarded version.
  fail_if_success
    have : Spec.Derivation R.assumptions
        [⟨[], lhsCode .addCommute, .real, none⟩]
        [⟨[], rhsCode .addCommute, .real, none⟩] := by
      solve_fp [admitted R .addCommute (by decide)]
  trivial

theorem output_cast_retained (B : Nat) (R : FloatDTypeAddFPEquiv.Rules) : True := by
  fail_if_success
    have : FloatDTypeAddFPEquiv.original B ≡[R] VectorAddFPEquiv.optimized B := by
      split_equiv
      all_goals solve_fp
  trivial

theorem active_mask_retained (n B : Nat) (R : FlatVectorAddFPEquiv.Rules) : True := by
  fail_if_success
    have : FlatVectorAddFPEquiv.original n B ≡[R] FlatVectorAddFPEquiv.optimized (n + 1) B := by
      split_equiv
      all_goals solve_fp
  trivial

-- The unchanged printer reports the atom used by the generated proof.
/-- info: FP assumptions used by imported_fp_rule:
---
info:   none
---
info: Strict model certificates used by imported_fp_rule:
---
info:   model: VeriTile.Software.strictFP32/ieeeValue/maxnum/documented-exp-special-values
    certificate: List.Forall₂.cons (FP.ScalarArithmetic.strict_certificate FP.ScalarArithmetic.Atom.addCommute rfl) List.Forall₂.nil -/
#guard_msgs in
#print_fp_assumptions imported_fp_rule

/-- info: FP assumptions used by two_fp_sites:
---
info:   none
---
info: Strict model certificates used by two_fp_sites:
---
info:   model: VeriTile.Software.strictFP32/ieeeValue/maxnum/documented-exp-special-values
    certificate: List.Forall₂.cons (FP.ScalarArithmetic.strict_certificate FP.ScalarArithmetic.Atom.mulCommute rfl) List.Forall₂.nil -/
#guard_msgs in
#print_fp_assumptions two_fp_sites

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Tests.EquivTactics

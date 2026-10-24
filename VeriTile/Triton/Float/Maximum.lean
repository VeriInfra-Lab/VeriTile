import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.GuardedStrict

/-! Scalar FP32 maximum rewrites certified in the fixed maxnum software profile.
The original report lookup remains available as historical metadata. These
four rules do not add experimental assumptions to the public rule set. -/
namespace VeriTile.Triton.FP.Maximum
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

inductive Atom where
  | max_commute | max_assoc | max_idem | max_neg_inf
  deriving DecidableEq, Repr

/-- Every max below is tl.maximum at fp32 precision. -/
def Atom.ruleID : Atom → String
  -- max(a, b) → max(b, a); finite a,b.
  | .max_commute => "MAX-COMMUTE"
  -- max(max(a, b), c) → max(a, max(b, c)); finite a,b,c.
  | .max_assoc => "MAX-ASSOC"
  -- max(a, a) → a; finite a.
  | .max_idem => "MAX-IDEM"
  -- max(-inf, a) → a; finite a. -inf is a literal, not a sampled input.
  | .max_neg_inf => "MAX-NEG-INF"

def candidates : List Atom := [.max_commute, .max_assoc, .max_idem, .max_neg_inf]

def Atom.guards : Atom → List OperandGuard
  | .max_commute => [⟨"a", .finite⟩, ⟨"b", .finite⟩]
  | .max_assoc => [⟨"a", .finite⟩, ⟨"b", .finite⟩, ⟨"c", .finite⟩]
  | .max_idem | .max_neg_inf => [⟨"a", .finite⟩]

/-- Retain the default floating maximum primitive used by tl.maximum.
The enclosing fragment supplies FP32 precision and finite operand guards.
The execution certificate below uses its explicit maxnum implementation. -/
def maximum (a b : Op .real []) : Op .real [] :=
  .maximumFloat .real .nil a b

def lhs (a : Atom) : GuardedFragment := ⟨a.guards, fragment (match a with
  | .max_commute => maximum (ref "a") (ref "b")
  | .max_assoc => maximum (maximum (ref "a") (ref "b")) (ref "c")
  | .max_idem => maximum (ref "a") (ref "a")
  | .max_neg_inf => maximum .negInf (ref "a")), .real, none⟩

def rhs (a : Atom) : GuardedFragment := ⟨a.guards, fragment (match a with
  | .max_commute => maximum (ref "b") (ref "a")
  | .max_assoc => maximum (ref "a") (maximum (ref "b") (ref "c"))
  | .max_idem | .max_neg_inf => ref "a"), .real, none⟩

/-- Match the exact relation, precision and operand domain before selecting a row. -/
def Atom.matches (a : Atom) (row : ReportedScalarRule) : Bool :=
  decide (row.report.ruleID = a.ruleID ∧ row.report.input = "fp32" ∧
    row.report.compute = "fp32" ∧ row.report.accumulator = "fp32" ∧
    row.report.output = "fp32" ∧ row.guards = a.guards)

/-- Historical accepted-row lookup, retained for existing callers. The strict
proof below does not consume the row or its evidence. -/
def Atom.report? (a : Atom) : Option ReportedScalarRule :=
  SupplementalAdmission.all.find? a.matches

abbrev Atom.Available (a : Atom) : Prop := a.report?.isSome = true

def report (a : Atom) (h : a.Available) : ReportedScalarRule := a.report?.get h

def entry (a : Atom) (h : a.Available) : Spec.RuleEntry GuardedFragment :=
  (report a h).report.bind [lhs a] [rhs a]

def Atom.entry? (a : Atom) : Option (Spec.RuleEntry GuardedFragment) :=
  a.report?.map fun row => row.report.bind [lhs a] [rhs a]

/-- Execute the actual fragments, including their precision, registers and
finite guards, before applying the shared raw-word maximum laws. -/
theorem strict_sound (a : Atom) : StrictFragmentSound (lhs a) (rhs a) := by
  refine ⟨rfl, ?_⟩
  intro rest s hg
  cases a with
  | max_commute =>
    obtain ⟨va, ha, hfa⟩ := hg ⟨"a", .finite⟩ (by simp [lhs, Atom.guards])
    obtain ⟨vb, hb, hfb⟩ := hg ⟨"b", .finite⟩ (by simp [lhs, Atom.guards])
    dsimp only [lhs, FloatDType.toTileDType, toFloat, strictDomain] at ha hb hfa hfb
    simp [lhs, rhs, fragment, fragmentAlgebra, maximum, ref, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      ha, hb, scalarOutput, State.setReg, Software.fp32Context, toFloat, ofFloat]
    funext i
    cases i
    exact MaximumExact.maximumValue_commute _ _ hfa hfb
  | max_assoc =>
    obtain ⟨va, ha, hfa⟩ := hg ⟨"a", .finite⟩ (by simp [lhs, Atom.guards])
    obtain ⟨vb, hb, hfb⟩ := hg ⟨"b", .finite⟩ (by simp [lhs, Atom.guards])
    obtain ⟨vc, hc, hfc⟩ := hg ⟨"c", .finite⟩ (by simp [lhs, Atom.guards])
    dsimp only [lhs, FloatDType.toTileDType, toFloat, strictDomain] at ha hb hc hfa hfb hfc
    simp [lhs, rhs, fragment, fragmentAlgebra, maximum, ref, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      ha, hb, hc, scalarOutput, State.setReg, Software.fp32Context, toFloat, ofFloat]
    funext i
    cases i
    exact MaximumExact.maximumValue_assoc _ _ _ hfa hfb hfc
  | max_idem =>
    obtain ⟨va, ha, hfa⟩ := hg ⟨"a", .finite⟩ (by simp [lhs, Atom.guards])
    dsimp only [lhs, FloatDType.toTileDType, toFloat, strictDomain] at ha hfa
    simp [lhs, rhs, fragment, fragmentAlgebra, maximum, ref, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      ha, scalarOutput, State.setReg, Software.fp32Context, toFloat, ofFloat]
    funext i
    cases i
    exact MaximumExact.maximumValue_idem _ hfa
  | max_neg_inf =>
    obtain ⟨va, ha, hfa⟩ := hg ⟨"a", .finite⟩ (by simp [lhs, Atom.guards])
    dsimp only [lhs, FloatDType.toTileDType, toFloat, strictDomain] at ha hfa
    simp [lhs, rhs, fragment, fragmentAlgebra, maximum, ref, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType,
      ha, scalarOutput, State.setReg, Software.strictFP32, Software.scalarFP32,
      toFloat, ofFloat, Software.fp32Context]
    funext i
    cases i
    exact MaximumExact.maximumValue_neg_inf _ hfa

/-- Compatibility handle: all four maximum laws now have strict certificates. -/
structure Rules where
  empty : True := True.intro

def Rules.assumptions (_ : Rules) : Spec.Assumptions GuardedFragment :=
  []

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- The old call shape now uses the fixed strict theory and no report axiom. -/
theorem admitted (R : Rules) (a : Atom) (_h : a.Available) :
    Spec.Derivation R.assumptions [lhs a] [rhs a] :=
  strict_derivation R.assumptions (strict_sound a)

theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [lhs a] ≡[R] [rhs a] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R a h)

/-- The preferred entry point has neither a report-selection nor an empirical
rule-set parameter. Its finite domain remains in the guarded fragments. -/
theorem strict_rewrite (a : Atom) :
    [lhs a] ≡[([] : Spec.Assumptions GuardedFragment)] [rhs a] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (strict_derivation [] (strict_sound a))

#axiomsClean strict_sound
#axiomsClean admitted
#axiomsClean strict_rewrite

end VeriTile.Triton.FP.Maximum

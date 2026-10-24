import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.ExpAdmission
import VeriTile.Triton.Float.GuardedStrict

/-! FP32 exponential candidates retain their exact intrinsic. Zero and
negative-infinity special values have strict execution certificates in the
documented profile. EXP-SUB remains a numerical assumption: the paired report
accepts only its libdevice variant. A report never equates the two primitives;
historical special-value reports are retained but not used as assumptions. -/
namespace VeriTile.Triton.FP.Exponential
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

inductive Backend where
  | tl | libdevice
  deriving DecidableEq, Repr

inductive Atom where
  | exp_sub (exp : Backend := .tl)
  | exp_zero (exp : Backend := .tl)
  | exp_neg_inf_sub (exp : Backend := .tl)
  deriving DecidableEq, Repr

/-- All operations compute in fp32. IDs remain stable across evidence kinds. -/
def Atom.ruleID : Atom → String
  -- exp(a - b) → exp(a) / exp(b); finite a,b; same backend on both sides.
  | .exp_sub .tl => "EXP-SUB-INTRINSIC"
  | .exp_sub .libdevice => "EXP-SUB"
  -- exp(0) → 1; no variable operands.
  | .exp_zero .tl => "EXP-ZERO"
  | .exp_zero .libdevice => "EXP-ZERO-LIBDEVICE"
  -- exp(-inf - a) → 0; finite a. -inf is a literal, not a sampled input.
  | .exp_neg_inf_sub .tl => "EXP-NEG-INF-SUB"
  | .exp_neg_inf_sub .libdevice => "EXP-NEG-INF-SUB-LIBDEVICE"

def candidates : List Atom := [.exp_sub .tl, .exp_sub .libdevice,
  .exp_zero .tl, .exp_zero .libdevice, .exp_neg_inf_sub .tl, .exp_neg_inf_sub .libdevice]

def Backend.exp : Backend → Op .real [] → Op .real []
  | .tl => .exp
  | .libdevice => .libdeviceExp

def guards : List OperandGuard := [⟨"a", .finite⟩, ⟨"b", .finite⟩]
def Atom.guards : Atom → List OperandGuard
  | .exp_sub _ => VeriTile.Triton.FP.Exponential.guards
  | .exp_zero _ => []
  | .exp_neg_inf_sub _ => [⟨"a", .finite⟩]

def Atom.lhs : Atom → GuardedFragment
  | a => ⟨a.guards, fragment (match a with
    | .exp_sub exp => exp.exp (minus (ref "a") (ref "b"))
    | .exp_zero exp => exp.exp (.const 0)
    | .exp_neg_inf_sub exp => exp.exp (minus .negInf (ref "a"))), .real, none⟩

def Atom.rhs : Atom → GuardedFragment
  | a => ⟨a.guards, fragment (match a with
    | .exp_sub exp => divide (exp.exp (ref "a")) (exp.exp (ref "b"))
    | .exp_zero _ => .const 1
    | .exp_neg_inf_sub _ => .const 0), .real, none⟩

/-- The original FP32 special-value fragments execute identically in the fixed
documented primitive profile. The proof includes their actual assignments. -/
theorem strict_exp_zero (backend : Backend) :
    StrictFragmentSound (Atom.exp_zero backend).lhs (Atom.exp_zero backend).rhs := by
  refine ⟨rfl, ?_⟩
  intro rest s _
  cases backend <;>
    simp [Atom.lhs, Atom.rhs, Backend.exp, fragment, fragmentAlgebra,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, Function.comp_def,
      Software.strictFP32, Software.scalarFP32, Software.fp32Context,
      ExponentialExact.tl_exp_zero, ExponentialExact.libdevice_exp_zero,
      ExponentialExact.positiveOne, scalarOutput, toFloat, State.setReg_same]

/-- The finite guard is read from the scalar register domain; it includes
signed zero and subnormal inputs, and excludes infinities and NaNs. -/
theorem strict_exp_neg_inf_sub (backend : Backend) :
    StrictFragmentSound (Atom.exp_neg_inf_sub backend).lhs
      (Atom.exp_neg_inf_sub backend).rhs := by
  refine ⟨rfl, ?_⟩
  intro rest s hd
  obtain ⟨v, hv, hf⟩ := hd ⟨"a", .finite⟩ (by simp [Atom.lhs, Atom.guards])
  change s.regs .real [] "a" = some v at hv
  have hfinite (i : TileIndex []) : ExponentialExact.finite (v i) := by
    cases i
    exact (ExponentialExact.finite_iff_toRat _).mpr hf
  cases backend <;>
    simp [Atom.lhs, Atom.rhs, Backend.exp, fragment, fragmentAlgebra, minus, ref,
      run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric,
      Software.strictFP32, Software.scalarFP32, Software.fp32Context,
      hv, bop, Function.comp_def, ExponentialExact.tl_exp_neg_inf_sub,
      ExponentialExact.libdevice_exp_neg_inf_sub, hfinite,
      scalarOutput, toFloat, State.setReg_same]

def Atom.isStrict : Atom → Bool
  | .exp_zero _ | .exp_neg_inf_sub _ => true
  | .exp_sub _ => false

theorem strict_sound (a : Atom) (h : a.isStrict = true) :
    StrictFragmentSound a.lhs a.rhs := by
  cases a with
  | exp_sub backend => simp [Atom.isStrict] at h
  | exp_zero backend => exact strict_exp_zero backend
  | exp_neg_inf_sub backend => exact strict_exp_neg_inf_sub backend

/-- Match the exact relation, precision and operand domain before selecting a row. -/
def Atom.matches (a : Atom) (row : ReportedScalarRule) : Bool :=
  decide (row.report.ruleID = a.ruleID ∧ row.report.input = "fp32" ∧
    row.report.compute = "fp32" ∧ row.report.accumulator = "fp32" ∧
    row.report.output = "fp32" ∧ row.guards = a.guards)

/-- Only accepted rows in the generated table can activate a candidate. -/
def Atom.report? (a : Atom) : Option ReportedScalarRule :=
  ExpAdmission.all.find? a.matches

abbrev Atom.Available (a : Atom) : Prop := (a.isStrict || a.report?.isSome) = true

abbrev Atom.EmpiricalAvailable (a : Atom) : Prop :=
  a.isStrict = false ∧ a.report?.isSome = true

def report (a : Atom) (h : a.report?.isSome = true) : ReportedScalarRule := a.report?.get h

def entry (a : Atom) (h : a.report?.isSome = true) : Spec.RuleEntry GuardedFragment :=
  (report a h).report.bind [Atom.lhs a] [Atom.rhs a]

def Atom.entry? (a : Atom) : Option (Spec.RuleEntry GuardedFragment) :=
  if a.isStrict then none
  else a.report?.map fun row => row.report.bind [Atom.lhs a] [Atom.rhs a]

theorem strict_entry_none (a : Atom) (h : a.isStrict = true) : a.entry? = none := by
  simp [Atom.entry?, h]

/-- External validation is required only for candidates selected by the report. -/
structure Rules where
  arithmetic : ScalarArithmetic.Rules
  validated : ∀ (a : Atom) (h : a.EmpiricalAvailable),
    Spec.EvidenceValidated (entry a h.2).rule (entry a h.2).evidence

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.arithmetic.assumptions ++ candidates.filterMap Atom.entry?

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- Strict candidates use their execution certificate. Only EXP-SUB enters
through a validated empirical assumption. -/
theorem admitted (R : Rules) (a : Atom) (h : a.Available) :
    Spec.Derivation R.assumptions [Atom.lhs a] [Atom.rhs a] := by
  by_cases hs : a.isStrict = true
  · exact strict_derivation _ (strict_sound a hs)
  have hs' : a.isStrict = false := Bool.eq_false_iff.mpr hs
  have havailable : a.report?.isSome = true := by simpa [Atom.Available, hs'] using h
  apply Spec.Derivation.atom (entry a havailable)
  · apply List.mem_append_right
    apply List.mem_filterMap.mpr
    refine ⟨a, ?_, ?_⟩
    · rcases a with ⟨_ | _⟩ | ⟨_ | _⟩ | ⟨_ | _⟩ <;> simp [candidates]
    · have available : a.report?.isSome = true := havailable
      cases hr : a.report? with
      | none => simp [hr] at available
      | some row => simp [Atom.entry?, hs', entry, report, hr]
  · exact (report a havailable).report.admit _ _ (R.validated a ⟨hs', havailable⟩)

theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [Atom.lhs a] ≡[R] [Atom.rhs a] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R a h)

theorem arithmetic_derivation (R : Rules) {lhs rhs : List GuardedFragment}
    (h : Spec.Derivation R.arithmetic.assumptions lhs rhs) :
    Spec.Derivation R.assumptions lhs rhs := by
  induction h with
  | refl code => exact .refl code
  | atom e he ha => exact .atom e (List.mem_append_left _ he) ha
  | strict certificate => exact .strict certificate
  | symm _ ih => exact .symm ih
  | trans _ _ ih₁ ih₂ => exact .trans ih₁ ih₂
  | frame beforeCode afterCode _ ih => exact .frame beforeCode afterCode ih

theorem arithmetic_models {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) :
    Models R.arithmetic.assumptions M D :=
  fun lhs rhs h => hM lhs rhs (arithmetic_derivation R h)

/-- The stable-softmax derivation specifically requires libdevice.exp. -/
def lhs : GuardedFragment := Atom.lhs (.exp_sub .libdevice)
def rhs : GuardedFragment := Atom.rhs (.exp_sub .libdevice)

#axiomsClean strict_exp_zero
#axiomsClean strict_exp_neg_inf_sub
#axiomsClean strict_sound
#axiomsClean strict_entry_none
#axiomsClean admitted

end VeriTile.Triton.FP.Exponential

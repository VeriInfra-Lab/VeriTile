import VeriTile.Triton.Float.ScalarArithmetic
import VeriTile.Triton.Float.GuardedStrict
import VeriTile.Triton.Float.CountConversionExact

/-! Strict count-conversion rules for the fixed FP32 software profile.
The count is a nonnegative signed-int32 value embedded in Nat. Exact proves
that the whole range, including its successor, fits signed int32, and proves
the selected conversion/addition identity using nearest-even without flushing.

The successor bound is retained in the fragment syntax: outside `i < 2^24`
both fragments return zero. These rules have concrete execution certificates;
only the separate arithmetic rules remain in the experimental assumption set.
Historical experiment records remain in CountAdmission, outside this module. -/
namespace VeriTile.Triton.FP.CountConversion
open Structural Guarded ScalarArithmetic
open scoped VeriTile.Spec

inductive Atom where
  | zero | successor
  deriving DecidableEq, Repr

def Atom.ruleID : Atom → String
  -- fp32(int32(0)) → fp32(0); no variable operands.
  | .zero => "COUNT-ZERO"
  -- fp32(i + 1) → fp32(i) + fp32(1); integer 0 ≤ i < 2^24.
  -- The increment is performed in int32; the addition on the right is fp32.
  | .successor => "COUNT-SUCCESSOR"

def candidates : List Atom := [.zero, .successor]

/-- The strict successor domain is part of the scalar syntax. The endpoint
also fits signed int32, as proved in `Exact.int32_ofNat_toInt`. -/
def limit : Nat := 16777216

def index : Op .nat [] := .ref .nat [] "i"
def bounded (e : Op .real []) : Op .real [] :=
  .where (.lt .nat .nil index (.constNat limit)) e (.const 0)

def lhsCode : Atom → List ComputeStmt
  | .zero => fragment (.natToReal (.constNat 0))
  | .successor => fragment (bounded (.natToReal (.add .nat .nil index (.constNat 1))))

def rhsCode : Atom → List ComputeStmt
  | .zero => fragment (.const 0)
  | .successor => fragment (bounded (plus (.natToReal index) (.const 1)))

def lhs (a : Atom) : GuardedFragment := ⟨[], lhsCode a, .real, none⟩
def rhs (a : Atom) : GuardedFragment := ⟨[], rhsCode a, .real, none⟩

set_option maxHeartbeats 1600000
set_option maxRecDepth 4000
private theorem run_zero_left {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    run M (lhsCode .zero) s =
      some (s.setReg "out" .real [] (fun _ => M.fromNat (some .fp32) 0)) := by
  simp [lhsCode, fragment, run, step, evalExpr, evalComputeOp, evalOp_unfold]

private theorem run_zero_right {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    run M (rhsCode .zero) s =
      some (s.setReg "out" .real [] (fun _ => M.literal (some .fp32) .real 0)) := by
  simp [rhsCode, fragment, run, step, evalExpr, evalComputeOp, evalOp_unfold]

private theorem run_successor_left {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    run M (lhsCode .successor) s =
      (s.regs .nat [] "i").map (fun v => s.setReg "out" .real []
        (fun _ => if v PUnit.unit < limit then M.fromNat (some .fp32) (v PUnit.unit + 1)
          else M.literal (some .fp32) .real 0)) := by
  cases h : s.regs .nat [] "i" <;>
    simp [lhsCode, fragment, bounded, index, run, step, evalExpr, evalComputeOp,
      evalOp_unfold, numeric, numericLt, natLt, bop, h]
  all_goals rfl

private theorem run_successor_right {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    run M (rhsCode .successor) s =
      (s.regs .nat [] "i").map (fun v => s.setReg "out" .real []
        (fun _ => if v PUnit.unit < limit then M.binary (some .fp32) .real .add
          (M.fromNat (some .fp32) (v PUnit.unit)) (M.literal (some .fp32) .real 1)
          else M.literal (some .fp32) .real 0)) := by
  cases h : s.regs .nat [] "i" <;>
    simp [rhsCode, fragment, bounded, index, plus, run, step, evalExpr, evalComputeOp,
      evalOp_unfold, numeric, numericLt, natLt, bop, h]
  all_goals rfl

private theorem strict_literal_zero (rest : Algebra (FP.Value .fp32)) :
    (Software.strictFP32 rest).literal (some .fp32) .real 0 = FP.Value.zero .fp32 := by
  simp [Software.strictFP32, Software.scalarFP32, Software.fp32Context]

private theorem strict_literal_one (rest : Algebra (FP.Value .fp32)) :
    (Software.strictFP32 rest).literal (some .fp32) .real 1 =
      FP.Value.ofBits .fp32 0x3f800000 := by
  simp [Software.strictFP32, Software.scalarFP32, Software.fp32Context]

/-- Execute both actual fragments in the fixed concrete software profile.
The equality includes the full resulting state, not only an abstract scalar
rewrite; it also preserves register-read failure on both sides. -/
theorem strict_run (a : Atom) (rest : Algebra (FP.Value .fp32))
    (s : State (FP.Value .fp32)) :
    run (Software.strictFP32 rest) (lhsCode a) s =
      run (Software.strictFP32 rest) (rhsCode a) s := by
  cases a with
  | zero =>
    rw [run_zero_left, run_zero_right]
    simp only [Software.strictFP32_fromNat, strict_literal_zero]
    exact congrArg (fun x => some (s.setReg "out" .real [] (fun _ => x))) Exact.count_zero
  | successor =>
    rw [run_successor_left, run_successor_right]
    apply congrArg (fun f => (s.regs .nat [] "i").map f)
    funext v
    apply congrArg (s.setReg "out" .real [])
    funext j
    by_cases hi : v PUnit.unit < limit
    · simp only [if_pos hi, strict_literal_one, Software.strictFP32_fromNat]
      rw [Software.strictFP32_add rest (some .fp32) .real _ _ rfl]
      have hc := Exact.count_successor (v PUnit.unit) hi
      rw [Exact.fromNat_one] at hc
      exact hc
    · simp only [if_neg hi]

/-- The strict certificate preserves the declared guard, output channel and
precision and compares actual execution in the fixed FP32 theory. -/
theorem strict_sound (a : Atom) : StrictFragmentSound (lhs a) (rhs a) := by
  refine ⟨rfl, ?_⟩
  intro rest s _
  simpa only [lhs, rhs, fragmentAlgebra] using
    congrArg (Option.map (scalarOutput .real)) (strict_run a rest s)

/-- Both count atoms are proved unconditionally in the selected model.
This compatibility predicate no longer depends on an experiment report. -/
abbrev Atom.Available (_a : Atom) : Prop := True

/-- Count contributes no experimental assumptions. Only arithmetic rules keep
an external validation obligation. -/
structure Rules where
  arithmetic : ScalarArithmetic.Rules

def Rules.assumptions (R : Rules) : Spec.Assumptions GuardedFragment :=
  R.arithmetic.assumptions

instance : CoeOut Rules (Spec.Assumptions GuardedFragment) := ⟨Rules.assumptions⟩

/-- A checked count step is available for any experimental assumption set. -/
@[spec_rule] theorem strict (R : Spec.Assumptions GuardedFragment) (a : Atom) :
    Spec.Derivation R [lhs a] [rhs a] :=
  Guarded.strict_derivation R (strict_sound a)

/-- Count rewrites already hold with an empty empirical rule set. -/
theorem strict_rewrite (a : Atom) :
    [lhs a] ≡[([] : Spec.Assumptions GuardedFragment)] [rhs a] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (strict [] a)

/-- Compatibility name: this uses a strict certificate, not empirical admission. -/
theorem admitted (R : Rules) (a : Atom) (_h : a.Available) :
    Spec.Derivation R.assumptions [lhs a] [rhs a] :=
  strict R.assumptions a

theorem rewrite (R : Rules) (a : Atom) (h : a.Available) : [lhs a] ≡[R] [rhs a] :=
  Spec.FloatingPoint.ofDerivation rfl trivial (admitted R a h)

theorem arithmetic_derivation (R : Rules) {lhs rhs : List GuardedFragment}
    (h : Spec.Derivation R.arithmetic.assumptions lhs rhs) :
    Spec.Derivation R.assumptions lhs rhs := h

theorem arithmetic_models {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) :
    Models R.arithmetic.assumptions M D := hM

#axiomsClean strict_run
#axiomsClean strict_sound
#axiomsClean strict
#axiomsClean strict_rewrite
#axiomsClean admitted
#axiomsClean rewrite

end VeriTile.Triton.FP.CountConversion

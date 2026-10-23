import VeriTile.Triton.Float.CountConversion
import VeriTile.Triton.Float.WelfordInduction

/-! Instantiate the strictly proved count rules for the recurrence proof.
The integer bound remains explicit in every successor application. Count
rules use concrete execution certificates and add no empirical assumptions. -/
namespace VeriTile.Triton.FP.CountConversion
open Structural Guarded ScalarArithmetic

set_option maxHeartbeats 1600000 in
theorem zero {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α) :
    M.fromNat (some .fp32) 0 = ScalarArithmetic.zero M := by
  have h := hM (lhs .zero) (rhs .zero) (admitted R .zero (by decide)) s
    (by simp [ScalarDomain, lhs]) (by simp [ScalarDomain, rhs])
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, ComputeDType.eraseDType] at h
  simp [State.setReg, ScalarArithmetic.zero] at h ⊢
  exact congrFun h PUnit.unit

set_option maxHeartbeats 1600000 in
theorem successor {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (i : Nat) (hi : i < limit) :
    M.fromNat (some .fp32) (i + 1) = ScalarArithmetic.add M (M.fromNat (some .fp32) i) (one M) := by
  let t := s.setReg "i" .nat [] (fun _ => i)
  have h := hM (lhs .successor) (rhs .successor) (admitted R .successor (by decide)) t
    (by simp [ScalarDomain, lhs]) (by simp [ScalarDomain, rhs])
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, bounded, index, plus, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType, t, State.setReg_same] at h
  simp [State.setReg, numeric, numericLt, natLt, bop, hi,
    ScalarArithmetic.add, ScalarArithmetic.one] at h ⊢
  exact congrFun h PUnit.unit

/-- Every conversion needed by the original N-step recurrence, with its
upper bound retained. Both count steps use strict scalar certificates. -/
theorem conversion {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (N : Nat) (hN : N ≤ limit) : WelfordInduction.CountConversion M N :=
  ⟨zero R M D hM s, fun i hi => successor R M D hM s i (lt_of_lt_of_le hi hN)⟩

/-- The concrete profile satisfies the complete bounded count recurrence
directly, with no experimental rule set or model-of-assumptions premise. -/
theorem conversion_strict (rest : Algebra (FP.Value .fp32))
    (N : Nat) (hN : N ≤ limit) :
    WelfordInduction.CountConversion (Software.strictFP32 rest) N := by
  constructor
  · simp only [ScalarArithmetic.zero, Software.strictFP32_fromNat]
    have hZero : (Software.strictFP32 rest).literal (some .fp32) .real 0 =
        FP.Value.zero .fp32 := by
      simp [Software.strictFP32, Software.scalarFP32, Software.fp32Context]
    rw [hZero]
    exact Exact.count_zero
  · intro i hi
    simp only [ScalarArithmetic.add, ScalarArithmetic.one, Software.strictFP32_fromNat]
    have hOne : (Software.strictFP32 rest).literal (some .fp32) .real 1 =
        FP.Value.ofBits .fp32 0x3f800000 := by
      simp [Software.strictFP32, Software.scalarFP32, Software.fp32Context]
    rw [hOne, Software.strictFP32_add rest (some .fp32) .real _ _ rfl]
    have hc := Exact.count_successor i (lt_of_lt_of_le hi hN)
    rw [Exact.fromNat_one] at hc
    exact hc

#axiomsClean zero
#axiomsClean successor
#axiomsClean conversion
#axiomsClean conversion_strict

end VeriTile.Triton.FP.CountConversion

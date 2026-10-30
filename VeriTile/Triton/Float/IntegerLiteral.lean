import VeriTile.Triton.Float.ScalarArithmetic

/-! The actual integer conversion and literal fragments agree in the fixed
strict FP32 software profile. Their certificate connects a source tensor
integer assignment to an inline literal without an empirical assumption or a
new input-domain restriction. Both sides round the same integer; no claim of
exact representability is needed. -/
namespace VeriTile.Triton.FP.IntegerLiteral
open Structural Guarded

def lhs (n : Int) : GuardedFragment :=
  ⟨[], ScalarArithmetic.fragment (.intToReal (.constInt n)), .real, none⟩

def rhs (n : Int) : GuardedFragment :=
  ⟨[], ScalarArithmetic.fragment (.const (n : ℝ)), .real, none⟩

theorem strict_certificate (n : Int) : StrictFragmentSound (lhs n) (rhs n) := by
  refine ⟨rfl, ?_⟩
  intro rest s _
  simp [lhs,rhs,ScalarArithmetic.fragment,fragmentAlgebra,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,scalarOutput,State.setReg,
    ComputeDType.eraseDType,Software.strictFP32,Software.fp32Context]

@[spec_rule] theorem strict (R : Spec.Assumptions GuardedFragment) (n : Int) :
    Spec.Derivation R [lhs n] [rhs n] :=
  Guarded.strict_derivation R (strict_certificate n)

theorem from_int_literal {α : Type} [Inhabited α] {R : Spec.Assumptions GuardedFragment}
    (M : Algebra α) (D : Domain α) (hM : Models R M D) (s : State α) (n : Int) :
    M.fromInt (some .fp32) n = M.literal (some .fp32) .real (n : ℝ) := by
  have hd : TypedScalarDomain D .real [] s := by
    intro g hg
    simp at hg
  have h := hM (lhs n) (rhs n) (strict R n) s hd hd
  simp [lhs,rhs,ScalarArithmetic.fragment,fragmentAlgebra,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,Option.bind,scalarOutput,State.setReg,
    ComputeDType.eraseDType] at h
  exact congrFun h PUnit.unit

#axiomsClean strict_certificate
#axiomsClean strict
#axiomsClean from_int_literal
end VeriTile.Triton.FP.IntegerLiteral

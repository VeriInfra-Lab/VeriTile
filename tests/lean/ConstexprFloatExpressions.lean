import VeriTile.Triton.DSL.ConstexprFloatExpressions
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
open VeriTile.Triton FP.Structural
namespace VeriTile.Tests.ConstexprFloatExpressions
noncomputable section
set_option linter.unusedSimpArgs false

def signature : CallSignature where
  scalars:=[("x",.fp32)]
  constexpr:=["C"]

def sample (compiled : ConstexprFloat.Model) (C : Nat) (x : ℝ) : ComputeKernel :=
  with_constexpr_float_model compiled in
  with_triton_call signature call_args [x := x] in triton {
    output = x * (C**-0.5 * 1.44269504)
  }

def factorTree (C : Nat) : ConstexprFloat.Expr :=
  .mul (.pow (.natural C) (.neg (.literal 0.5))) (.literal 1.44269504)

/-- Every constexpr primitive stays outside runtime tensor arithmetic. -/
theorem primitive_interpretation {α : Type} [Inhabited α] (M : Algebra α) (compiled : ConstexprFloat.Model)
    (C : Nat) (x : ℝ) (s : State α) :
    ∃ t,exec M (sample compiled C x) s=some t ∧
      t.regs .real [] "output"=some (fun _=>M.binary (some .fp32) .real .mul
        (M.literal (some .fp32) .real x)
        (M.literal (some .fp32) .real (ConstexprFloat.eval compiled (factorTree C)))) := by
  simp [sample,exec,ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ofFloat,toFloat,factorTree,bop,numeric]
  rfl

/-- error: floating power requires an immutable declared constexpr expression -/
#guard_msgs in
def runtimePower (compiled : ConstexprFloat.Model) (C : Nat) (x : ℝ) : ComputeKernel :=
  with_constexpr_float_model compiled in
  with_triton_call signature call_args [x := x] in triton {
    output = x ** -0.5
  }

/-- error: integer constexpr power is outside the floating-expression subset -/
#guard_msgs in
def integerPower (compiled : ConstexprFloat.Model) (C : Nat) (x : ℝ) : ComputeKernel :=
  with_constexpr_float_model compiled in
  with_triton_call signature call_args [x := x] in triton {
    output = x * (C ** 2)
  }

/-- error: runtime assignment shadows an integer constexpr -/
#guard_msgs in
def mutatedArgument (compiled : ConstexprFloat.Model) (C : Nat) (x : ℝ) : ComputeKernel :=
  with_constexpr_float_model compiled in
  with_triton_call signature call_args [x := x] in triton {
    C = 4
    output = x * (C ** -0.5)
  }

/-- The exact Real interpretation is separate from any rounded model. -/
theorem real_denotation (C : Nat) :
    ConstexprFloat.eval ConstexprFloat.real (factorTree C) =
      Real.rpow (C : ℝ) (-(0.5 : ℝ)) * 1.44269504 := rfl

#axiomsClean primitive_interpretation
#axiomsClean real_denotation
end
end VeriTile.Tests.ConstexprFloatExpressions

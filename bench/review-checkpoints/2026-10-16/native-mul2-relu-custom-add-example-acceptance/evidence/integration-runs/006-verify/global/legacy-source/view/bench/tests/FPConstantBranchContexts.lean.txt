import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! When both arms change, a literal condition selects the required guard.
The numerical law is still required at the actual prefix of that arm. -/
namespace FPConstantBranchContextsTests
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

private def x : Op .real [] := .ref .real [] "x"
private def one : Op .real [] := .const 1
private def padding : ComputeStmt := .assign .nat [] "padding" (.alg (.constNat 17))
private def body (swapped : Bool) : List ComputeStmt :=
  [padding, .assign .real [] "x" (.compute (.alg .fp32
    (if swapped then .add .real .nil one x else .add .real .nil x one)))]
private def inner : Site := .addFP32 [padding] [] x one

private def program (selected swapped : Bool) : Program where
  kernel := .mk [] [] [padding, .ifThenElse (.constBool selected)
    (body swapped) (body swapped ++ [padding]), padding]
  domain := [.branch [padding] (.constBool selected) selected inner]

theorem selected_arm (selected : Bool) (R : FP.ScalarArithmetic.Rules) :
    program selected false ≡[R] program selected true := by
  cases selected <;> equiv_decompose <;> fp_prove

set_option linter.unusedVariables false in
example (R : FP.ScalarArithmetic.Rules) : True := by
  fail_if_success
    have : ({ program true false with domain := [] } : Program) ≡[R]
        ({ program true true with domain := [] } : Program) := by
      equiv_decompose <;> fp_prove
  fail_if_success
    have : ({ program false false with domain :=
        [.branch [padding] (.constBool false) true inner] } : Program) ≡[R]
        ({ program false true with domain :=
        [.branch [padding] (.constBool false) true inner] } : Program) := by
      equiv_decompose <;> fp_prove
  fail_if_success
    have : ({ program true false with domain :=
        [.branch [] (.constBool true) true inner] } : Program) ≡[R]
        ({ program true true with domain :=
        [.branch [] (.constBool true) true inner] } : Program) := by
      equiv_decompose <;> fp_prove
  trivial

example : True := by
  fail_if_success
    have : program true false ≡[([] : Spec.Assumptions FP.GuardedFragment)]
        program true true := by
      equiv_decompose <;> fp_prove
  trivial

#axiomsClean contextual_constant_then
#axiomsClean contextual_constant_else
#axiomsClean selected_arm
end FPConstantBranchContextsTests

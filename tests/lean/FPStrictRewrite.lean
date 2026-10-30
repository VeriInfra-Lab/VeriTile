import VeriTile.Triton.Float.StrictRewrite
import VeriTile.Triton.Float.RewriteTactics

namespace FPStrictRewriteTests
open VeriTile Triton FP Structural
open GuardedRewrite (sameShape assignment)

variable [Inhabited (Value .fp32)]

-- Registration closes concrete contextual goals with no rule table or
-- domain hypothesis, including arbitrary configuration and source context.
theorem explicit_add (rest : Algebra (Value .fp32)) (cfg : Config)
    (s : State (Value .fp32)) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape) :
    GuardedRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (assignment .add out a b) (assignment .add out b a) := by
  fp_prove

theorem profiled_explicit_mul (rest : Algebra (Value .fp32)) (cfg : Config)
    (s : State (Value .fp32)) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape) :
    ProfiledRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (assignment .mul out a b) (assignment .mul out b a) := by
  fp_prove

theorem default_add (rest : Algebra (Value .fp32)) (cfg : Config)
    (s : State (Value .fp32)) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape) :
    ProfiledRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (.assign .real shape out (.alg (.add .real (sameShape shape) a b)))
      (.assign .real shape out (.alg (.add .real (sameShape shape) b a))) := by
  fp_prove

-- A successful original execution returns the same complete state. Prefix
-- support, loads, stores and any other context obligations are retained.
theorem successful_execution (rest : Algebra (Value .fp32)) (cfg : Config)
    (s result : State (Value .fp32)) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape)
    (success : run (Software.scalarFP32 rest cfg)
      (before ++ assignment .add out a b :: after) s = some result) :
    run (Software.scalarFP32 rest cfg)
      (before ++ assignment .add out b a :: after) s = some result := by
  rw [← explicit_add rest cfg s before after out a b]
  exact success

set_option linter.unusedVariables false in
example (rest : Algebra (Value .fp32)) (cfg : Config)
    (s : State (Value .fp32)) (before after : List ComputeStmt)
    (out : RegName) (a b : Op .real shape) : True := by
  -- An explicit FP64 operation remains outside this FP32 adapter.
  fail_if_success
    have : ProfiledRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
        (.assign .real shape out (.compute (.alg .fp64 (.add .real (sameShape shape) a b))))
        (.assign .real shape out (.compute (.alg .fp64 (.add .real (sameShape shape) b a)))) := by
      fp_prove
  -- Untagged .real operations require the declared default precision.
  fail_if_success
    have : GuardedRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
        (.assign .real shape out (.alg (.add .real (sameShape shape) a b)))
        (.assign .real shape out (.alg (.add .real (sameShape shape) b a))) := by
      fp_prove
  -- No identity is inferred for the arbitrary residual algebra.
  fail_if_success
    have : GuardedRewrite.Contextual rest s before after
        (assignment .add out a b) (assignment .add out b a) := by
      fp_prove
  trivial

#guard_msgs (drop info) in
#axiomsClean explicit_add
#guard_msgs (drop info) in
#axiomsClean profiled_explicit_mul
#guard_msgs (drop info) in
#axiomsClean default_add
#guard_msgs (drop info) in
#axiomsClean successful_execution

end FPStrictRewriteTests

namespace FPStrictProgramTests
open VeriTile Triton FP Structural
open scoped VeriTile.Spec

private def operand (name : RegName) : Op .real [4] := .ref .real [4] name
private def read (name : RegName) : ComputeStmt :=
  .assign .real [4] name (.alg (.load .real (.region name (.arange 4)) .none))
private def readPrefix : List ComputeStmt := [read "x", read "y"]
private def write : ComputeStmt := .store .real [4] (.region "out" (.arange 4))
  (.alg (operand "out")) .none
private def explicitAdd (swapped : Bool) : ComputeStmt :=
  GuardedRewrite.assignment .add "out"
    (operand (if swapped then "y" else "x"))
    (operand (if swapped then "x" else "y"))
private def defaultAdd (swapped : Bool) : ComputeStmt :=
  .assign .real [4] "out" (.alg (.add .real (GuardedRewrite.sameShape [4])
    (operand (if swapped then "y" else "x"))
    (operand (if swapped then "x" else "y"))))

private def guardedProgram (swapped : Bool) : GuardedRewrite.Program :=
  ⟨.mk ["x", "y"] ["out"] (readPrefix ++ [explicitAdd swapped, write]),
    [⟨readPrefix, [4], operand "x", operand "y"⟩]⟩

private def profiledProgram (swapped : Bool) : ProfiledRewrite.Program :=
  ⟨.mk ["x", "y"] ["out"] (readPrefix ++ [defaultAdd swapped, write]),
    [.add readPrefix [4] (operand "x") (operand "y")]⟩

-- Public implementation equivalence retains its source syntax and domain;
-- strictly proved commutation needs no empirical entries in the rule table.
theorem guarded_empty_rules :
    guardedProgram false ≡[([] : Spec.Assumptions GuardedFragment)] guardedProgram true := by
  equiv_decompose
  all_goals fp_prove

theorem profiled_empty_rules :
    profiledProgram false ≡[([] : Spec.Assumptions GuardedFragment)] profiledProgram true := by
  equiv_decompose
  all_goals fp_prove

private def division (reciprocal : Bool) : ComputeStmt :=
  .assign .real [4] "out" (.alg (if reciprocal then
    .mul .real .scalarR (operand "out") (.div .real .nil (.const 1) (.const 2))
    else .div .real .scalarR (operand "out") (.const 2)))

private def mixedProgram (swapped reciprocal : Bool) : ProfiledRewrite.Program :=
  ⟨.mk ["x", "y"] ["out"] (readPrefix ++ [defaultAdd swapped, division reciprocal, write]),
    [.add readPrefix [4] (operand "x") (operand "y"),
     .div (readPrefix ++ [defaultAdd false]) 4 [] (operand "out") 2]⟩

-- Mixed proofs still use the original actual-operand guards. Only the
-- reciprocal transformation requires the experimental rule bundle.
theorem mixed_strict_experimental (R : ScalarArithmetic.Rules) :
    mixedProgram false false ≡[R] mixedProgram true true := by
  equiv_decompose (reverse := true)
  all_goals fp_prove

#guard_msgs (drop info) in
#axiomsClean guarded_empty_rules
#guard_msgs (drop info) in
#axiomsClean profiled_empty_rules
#guard_msgs (drop info) in
#axiomsClean mixed_strict_experimental

end FPStrictProgramTests

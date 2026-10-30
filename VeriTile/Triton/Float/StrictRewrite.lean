import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Triton.Float.ScalarBackend
import VeriTile.Triton.Float.ScalarOpsLaws

/-! Strict contextual rewrites for the concrete scalar FP32 software model.
The numerical proof is shared in `ScalarOpsLaws`; this module proves its
connection to operand evaluation, assignments, and the unchanged source
prefix/suffix. No experimental rule, finite-domain assumption, or arithmetic
law for an arbitrary algebra is introduced. The configuration stays explicit.
-/
namespace VeriTile.Triton.FP.StrictRewrite
open Structural
open GuardedRewrite (sameShape sameShape_left sameShape_right)

variable [Inhabited (Value .fp32)]

private theorem eval_add_commute (M : Algebra (Value .fp32))
    (cfg : Config) (p : Option ComputeDType)
    (binding : M.binary p .real .add = FP.add cfg)
    (a b : Op .real shape) (s : State (Value .fp32)) :
    Structural.evalOp M p (.add .real (sameShape shape) a b) s =
      Structural.evalOp M p (.add .real (sameShape shape) b a) s := by
  change ((Structural.evalOp M p a s).bind fun va => (Structural.evalOp M p b s).bind fun vb =>
      some (bop (M.binary p .real .add) (sameShape shape) va vb)) =
    ((Structural.evalOp M p b s).bind fun vb => (Structural.evalOp M p a s).bind fun va =>
      some (bop (M.binary p .real .add) (sameShape shape) vb va))
  cases ha : Structural.evalOp M p a s <;> cases hb : Structural.evalOp M p b s
  all_goals simp only [Option.bind_none, Option.bind_some]
  rename_i va vb
  apply congrArg some
  funext i
  simp only [bop, sameShape_left, sameShape_right, binding]
  exact ScalarOpsLaws.add_comm cfg _ _

private theorem eval_mul_commute (M : Algebra (Value .fp32))
    (cfg : Config) (p : Option ComputeDType)
    (binding : M.binary p .real .mul = FP.mul cfg)
    (a b : Op .real shape) (s : State (Value .fp32)) :
    Structural.evalOp M p (.mul .real (sameShape shape) a b) s =
      Structural.evalOp M p (.mul .real (sameShape shape) b a) s := by
  change ((Structural.evalOp M p a s).bind fun va => (Structural.evalOp M p b s).bind fun vb =>
      some (bop (M.binary p .real .mul) (sameShape shape) va vb)) =
    ((Structural.evalOp M p b s).bind fun vb => (Structural.evalOp M p a s).bind fun va =>
      some (bop (M.binary p .real .mul) (sameShape shape) vb va))
  cases ha : Structural.evalOp M p a s <;> cases hb : Structural.evalOp M p b s
  all_goals simp only [Option.bind_none, Option.bind_some]
  rename_i va vb
  apply congrArg some
  funext i
  simp only [bop, sameShape_left, sameShape_right, binding]
  exact ScalarOpsLaws.mul_comm cfg _ _

omit [Inhabited (Value .fp32)] in
private theorem run_assignment_congr {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (out : RegName)
    (lhs rhs : ComputeExpr dtype shape)
    (h : ∀ t, evalExpr M lhs t = evalExpr M rhs t) :
    run M (before ++ .assign dtype shape out lhs :: after) s =
      run M (before ++ .assign dtype shape out rhs :: after) s := by
  rw [run_append, run_append]
  cases hp : run M before s with
  | none => rfl
  | some t =>
    simp only [Option.bind_some, run, step]
    rw [h t]

/-- Explicit FP32 addition, including all NaNs, zero signs, and flush modes. -/
@[spec_rule] theorem guarded_add_commute
    (rest : Algebra (Value .fp32)) (cfg : Config) (s : State (Value .fp32))
    (before after : List ComputeStmt) (out : RegName) (a b : Op .real shape) :
    GuardedRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (GuardedRewrite.assignment .add out a b) (GuardedRewrite.assignment .add out b a) := by
  apply run_assignment_congr
  intro t
  exact eval_add_commute _ cfg (some .fp32) rfl a b t

/-- Explicit FP32 multiplication in the unchanged complete execution context. -/
@[spec_rule] theorem guarded_mul_commute
    (rest : Algebra (Value .fp32)) (cfg : Config) (s : State (Value .fp32))
    (before after : List ComputeStmt) (out : RegName) (a b : Op .real shape) :
    GuardedRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (GuardedRewrite.assignment .mul out a b) (GuardedRewrite.assignment .mul out b a) := by
  apply run_assignment_congr
  intro t
  exact eval_mul_commute _ cfg (some .fp32) rfl a b t

/-- Explicit precision is preserved when the surrounding source has an FP32 default. -/
@[spec_rule] theorem profiled_add_fp32_commute
    (rest : Algebra (Value .fp32)) (cfg : Config) (s : State (Value .fp32))
    (before after : List ComputeStmt) (out : RegName) (a b : Op .real shape) :
    ProfiledRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (GuardedRewrite.assignment .add out a b) (GuardedRewrite.assignment .add out b a) := by
  apply run_assignment_congr
  intro t
  exact eval_add_commute _ cfg (some .fp32) rfl a b t

@[spec_rule] theorem profiled_mul_fp32_commute
    (rest : Algebra (Value .fp32)) (cfg : Config) (s : State (Value .fp32))
    (before after : List ComputeStmt) (out : RegName) (a b : Op .real shape) :
    ProfiledRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (GuardedRewrite.assignment .mul out a b) (GuardedRewrite.assignment .mul out b a) := by
  apply run_assignment_congr
  intro t
  exact eval_mul_commute _ cfg (some .fp32) rfl a b t

/-- An untagged operation uses the declared FP32 profile, not Real arithmetic. -/
@[spec_rule] theorem profiled_add_commute
    (rest : Algebra (Value .fp32)) (cfg : Config) (s : State (Value .fp32))
    (before after : List ComputeStmt) (out : RegName) (a b : Op .real shape) :
    ProfiledRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (.assign .real shape out (.alg (.add .real (sameShape shape) a b)))
      (.assign .real shape out (.alg (.add .real (sameShape shape) b a))) := by
  apply run_assignment_congr
  intro t
  exact eval_add_commute _ cfg none rfl a b t

@[spec_rule] theorem profiled_mul_commute
    (rest : Algebra (Value .fp32)) (cfg : Config) (s : State (Value .fp32))
    (before after : List ComputeStmt) (out : RegName) (a b : Op .real shape) :
    ProfiledRewrite.Contextual (Software.scalarFP32 rest cfg) s before after
      (.assign .real shape out (.alg (.mul .real (sameShape shape) a b)))
      (.assign .real shape out (.alg (.mul .real (sameShape shape) b a))) := by
  apply run_assignment_congr
  intro t
  exact eval_mul_commute _ cfg none rfl a b t

#axiomsClean guarded_add_commute
#axiomsClean guarded_mul_commute
#axiomsClean profiled_add_fp32_commute
#axiomsClean profiled_mul_fp32_commute
#axiomsClean profiled_add_commute
#axiomsClean profiled_mul_commute

end VeriTile.Triton.FP.StrictRewrite

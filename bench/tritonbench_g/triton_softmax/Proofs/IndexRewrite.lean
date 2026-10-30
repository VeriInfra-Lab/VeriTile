import VeriTile.Triton.Float.ExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.TritonSoftmax.IndexRewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

@[spec_rule] theorem index_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .nil a b) (.mul .nat .nil b a) := by
  intro t _
  change (evalOp (engine M) none a t >>= fun x =>
    evalOp (engine M) none b t >>= fun y => some (bop Nat.mul .nil x y)) =
    (evalOp (engine M) none b t >>= fun y =>
    evalOp (engine M) none a t >>= fun x => some (bop Nat.mul .nil y x))
  cases ha : evalOp (engine M) none a t <;>
    cases hb : evalOp (engine M) none b t
  all_goals first
    | rfl
    | apply congrArg some
      funext i
      exact Nat.mul_comm _ _

/-- Equal address expressions preserve pointer evaluation, including failure. -/
@[spec_rule] theorem op_pointer_add {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (bc : Broadcast sa sb sh)
    (p p' : Op .ptr sa) (a a' : Op .nat sb)
    (hp : OpEquivalent M s before p p') (ha : OpEquivalent M s before a a') :
    OpEquivalent M s before (.ptrAdd bc p a) (.ptrAdd bc p' a') := by
  intro t ht
  simp only [FP.Structural.evalOp]
  rw [hp t ht, ha t ht]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax.IndexRewrite

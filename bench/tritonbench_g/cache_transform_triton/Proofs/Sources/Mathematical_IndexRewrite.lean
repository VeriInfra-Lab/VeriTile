import VeriTile.Triton.Float.IndexExpressionRewrite
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.CacheTransformTriton.IndexRewrite
open VeriTile.Triton FP.Structural FP.ProfiledRewrite

@[spec_rule] theorem index_product_commutes {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before : List ComputeStmt) (a b : Op .nat []) :
    OpEquivalent M s before (.mul .nat .nil a b) (.mul .nat .nil b a) :=
  FP.ProfiledRewrite.op_nat_mul_comm M s before a b

/-- The tactic composes the exact integer law through a vector index addition.
No floating law, finite-value condition or experiment is used. -/
theorem index_assignment {α : Type} [Inhabited α] (M : Algebra α)
    (s : State α) (before after : List ComputeStmt) (B : Nat) :
    Contextual M s before after
      (.assign .nat [B] "idx" (.alg (.add .nat .scalarL
        (.mul .nat .nil (.programId 0) (.constNat B)) (.arange B))))
      (.assign .nat [B] "idx" (.alg (.add .nat .scalarL
        (.mul .nat .nil (.constNat B) (.programId 0)) (.arange B)))) := by
  solve_fp

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.CacheTransformTriton.IndexRewrite

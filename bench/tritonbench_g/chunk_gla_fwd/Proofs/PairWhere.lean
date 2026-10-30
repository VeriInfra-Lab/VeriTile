import VeriTile.Triton.Float.ProfiledRewrite
import VeriTile.Meta.StatementAudit

/-! Task-local connection of the existing strict multiplication certificate to
the causal where branch in both selected diagonal sources. No scalar FP law
is re-proved; every lane keeps the existing reached-operand guards. -/
namespace VeriTile.Bench.Repairs.ChunkGlaFwd.PairWhere
open VeriTile Triton FP.Structural FP.Guarded FP.ProfiledRewrite
set_option maxHeartbeats 500000
set_option maxRecDepth 10000

/-- The same guarded rewrite inside an explicit fp32 computation. Operand
guards use the source computation context, including failed evaluations. -/
@[spec_rule] theorem selected_where_mul_fp32_scalar {α : Type} [Inhabited α]
    {R : Spec.Assumptions FP.GuardedFragment} (M : Algebra α) (D : Domain α)
    (hM : Models R M D) (s : State α) (before after : List ComputeStmt)
    (out : RegName) (a : Op .real (n :: rest)) (b : Op .real [])
    (condition : Op .bool (n :: rest)) (otherwise : Op .real (n :: rest))
    (hd : (Site.mulScalarFP32 before n rest a b).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.compute (.alg .fp32 (.where condition (.mul .real .scalarR a b) otherwise))))
      (.assign .real (n :: rest) out (.compute (.alg .fp32 (.where condition (.mul .real .scalarL b a) otherwise)))) := by
  unfold Contextual
  rw [run_append,run_append]
  cases hp : run (engine M) before s with
  | none => rfl
  | some t =>
    have hm : FP.Structural.evalOp (engine M) (some .fp32) (.mul .real .scalarR a b) t =
        FP.Structural.evalOp (engine M) (some .fp32) (.mul .real .scalarL b a) t := by
      change ((FP.Structural.evalOp (engine M) (some .fp32) a t).bind fun va =>
        (FP.Structural.evalOp (engine M) (some .fp32) b t).bind fun vb =>
          some (bop ((engine M).binary (some .fp32) .real .mul) .scalarR va vb)) =
        ((FP.Structural.evalOp (engine M) (some .fp32) b t).bind fun vb =>
        (FP.Structural.evalOp (engine M) (some .fp32) a t).bind fun va =>
          some (bop ((engine M).binary (some .fp32) .real .mul) .scalarL vb va))
      cases ha : FP.Structural.evalOp (engine M) (some .fp32) a t <;>
        cases hb : FP.Structural.evalOp (engine M) (some .fp32) b t
      all_goals simp only [Option.bind_none,Option.bind_some]
      rename_i va vb
      apply congrArg some
      funext i
      exact FP.ScalarArithmetic.mul_comm_strict M D hM t _ _ ((hd t hp va vb ha hb).1 i)
        (hd t hp va vb ha hb).2
    have hw : FP.Structural.evalOp (engine M) (some .fp32)
        (.where condition (.mul .real .scalarR a b) otherwise) t =
        FP.Structural.evalOp (engine M) (some .fp32)
        (.where condition (.mul .real .scalarL b a) otherwise) t := by
      change ((FP.Structural.evalOp (engine M) (some .fp32) condition t).bind fun vc =>
        (FP.Structural.evalOp (engine M) (some .fp32) (.mul .real .scalarR a b) t).bind fun va =>
        (FP.Structural.evalOp (engine M) (some .fp32) otherwise t).bind fun vb =>
          some (fun i => if vc i then va i else vb i)) = _
      rw [hm]
      rfl
    simp only [Option.bind_some,run,step,FP.Structural.evalExpr,evalComputeOp]
    erw [hw]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGlaFwd.PairWhere

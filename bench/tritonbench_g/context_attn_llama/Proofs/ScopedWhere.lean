import VeriTile.Triton.Float.TypedWhereRewrite
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.ScopedWhere
open VeriTile.Triton FP.Structural FP.Guarded FP.ProfiledRewrite
set_option maxHeartbeats 1000000
set_option maxRecDepth 20000
@[spec_rule] theorem typed {α : Type} [Inhabited α]
    (R : FP.TypedScalarArithmetic.Rules) (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out name : RegName) (n : Nat) (rest : TileShape) (scale : ℝ)
    (condition : Op .bool (n :: rest)) (otherwise : Op .real (n :: rest))
    (hd : (Site.mulScalarTypedFP32 before n rest (.ref .fp32 (n::rest) name) (.constFloat .fp32 scale)).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.where condition
        (.withPrecision .real .fp32 (.castFloat .fp32 .real (.mul .fp32 .scalarR (.ref .fp32 (n::rest) name) (.constFloat .fp32 scale)))) otherwise)))
      (.assign .real (n :: rest) out (.alg (.where condition
        (.withPrecision .real .fp32 (.castFloat .fp32 .real (.mul .fp32 .scalarL (.constFloat .fp32 scale) (.ref .fp32 (n::rest) name)))) otherwise))) := by
  have h := contextual_where_mul_typed_scalar R M D hM s before after out
    (.ref .fp32 (n::rest) name) (.constFloat .fp32 scale) condition otherwise hd
  unfold Contextual at h ⊢
  rw [run_append, run_append] at h ⊢
  cases hp : run (engine M) before s with
  | none => simp [hp]
  | some t =>
    simp only [hp, Option.bind_some] at h ⊢
    simpa only [run, step, evalExpr] using h
@[spec_rule] theorem real {α : Type} [Inhabited α]
    (R : FP.TypedScalarArithmetic.Rules) (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out name : RegName) (n : Nat) (rest : TileShape) (scale : ℝ)
    (condition : Op .bool (n :: rest)) (otherwise : Op .real (n :: rest))
    (hd : (Site.mulScalar before n rest (.ref .real (n::rest) name) (.const scale)).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.alg (.where condition
        (.withPrecision .real .fp32 (.mul .real .scalarR (.ref .real (n::rest) name) (.const scale))) otherwise)))
      (.assign .real (n :: rest) out (.alg (.where condition
        (.withPrecision .real .fp32 (.mul .real .scalarL (.const scale) (.ref .real (n::rest) name))) otherwise))) := by
  have h := contextual_where_mul_combined_scalar R M D hM s before after out
    (.ref .real (n::rest) name) (.const scale) condition otherwise hd
  unfold Contextual at h ⊢
  rw [run_append, run_append] at h ⊢
  cases hp : run (engine M) before s with
  | none => simp [hp]
  | some t =>
    simp only [hp, Option.bind_some] at h ⊢
    simpa only [run, step, evalExpr] using h
@[spec_rule] theorem real_compute {α : Type} [Inhabited α]
    (R : FP.TypedScalarArithmetic.Rules) (M : Algebra α) (D : FP.Guarded.Domain α)
    (hM : Models R.assumptions M D) (s : State α) (before after : List ComputeStmt)
    (out name : RegName) (n : Nat) (rest : TileShape) (scale : ℝ)
    (condition : Op .bool (n :: rest)) (otherwise : Op .real (n :: rest))
    (hd : (Site.mulScalar before n rest (.ref .real (n::rest) name) (.const scale)).Holds M D s) :
    Contextual M s before after
      (.assign .real (n :: rest) out (.compute (.alg .fp32 (.where condition
        (.mul .real .scalarR (.ref .real (n::rest) name) (.const scale)) otherwise))))
      (.assign .real (n :: rest) out (.compute (.alg .fp32 (.where condition
        (.mul .real .scalarL (.const scale) (.ref .real (n::rest) name)) otherwise)))) := by
  apply contextual_where_mul_fp32_scalar R.scalar M D
    (FP.TypedScalarArithmetic.scalar_models R M D hM) s before after out
    (.ref .real (n::rest) name) (.const scale) condition otherwise
  exact hd
#axiomsClean real_compute
#axiomsClean typed
#axiomsClean real
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.ScopedWhere

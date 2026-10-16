import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericExecution

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical

def unrestricted : FP.Guarded.Domain (WithBot ℝ) := fun _ _ => True

private theorem empty_derivation {σ : Type} {lhs rhs : List σ}
    (h : VeriTile.Spec.Derivation [] lhs rhs) : lhs = rhs := by
  induction h with
  | refl => rfl
  | atom entry member _ => cases member
  | symm _ ih => exact ih.symm
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂
  | frame before after _ ih => rw [ih]

private theorem empty_models : FP.Guarded.Models FPProof.assumptions model unrestricted := by
  intro lhs rhs h s _ _ a b ha hb
  have heq : lhs = rhs := List.singleton_inj.mp (empty_derivation h)
  subst rhs
  have hab : a = b := Option.some.inj (ha.symm.trans hb)
  subst b
  rfl

/-- The same successful numerical state is reached by the source candidate,
using complete typed FP equivalence, with no erasure or artificial arithmetic. -/
theorem both_execution (storage : TypedSource.Storage) (v : Bool) :
    run (engine model) (sample.source storage v).surfaceBody (initial storage) =
      some (numeric_run storage).val := by
  cases v
  case «false» => exact (numeric_run storage).property
  case «true» =>
    have h := FP.ProfiledRewrite.sound (FPProof.source_equivalence sample storage)
    have he := h (WithBot ℝ) model unrestricted empty_models (initial storage)
      (by intro site hs; cases hs)
    exact he.symm.trans (numeric_run storage).property

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution

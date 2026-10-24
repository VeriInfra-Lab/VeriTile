import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedExecution
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardFP
import VeriTile.Triton.Float.ProfiledSoundness
import VeriTile.Triton.Float.Countermodel

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

/-- Candidate execution is transported through the checked, assumption-free
whole-program FP equivalence, for the same successful initial state. -/
theorem candidate_runs (storage : Typed.Storage) (log causal fp16 : Bool) :
    run (engine model)
      (Typed.homogeneousKernel storage Bool.true "D" "G" "O"
        11 7 13 5 17 3 3 4 log causal fp16).surfaceBody
      (initial storage log) = some (source_run storage log causal fp16 Bool.false).val := by
  have h := FP.ProfiledRewrite.sound
    (Floating.backward_index_equiv storage "D" "G" "O"
      11 7 13 5 17 3 3 4 log causal fp16)
    (WithBot ℝ) model (fun _ _ => True)
    (by apply FP.models_of_atoms; intro e he; cases he)
    (initial storage log) (by simp [Floating.program])
  exact h.symm.trans (source_run storage log causal fp16 Bool.false).property

/-- All state components, including typed cells and untouched addresses,
agree with the original execution. No witness implementation is unfolded. -/
theorem variant_state (storage : Typed.Storage) (log causal fp16 v : Bool) :
    (source_run storage log causal fp16 v).val =
      (source_run storage log causal fp16 Bool.false).val := by
  cases v
  · rfl
  · exact Option.some.inj
      ((source_run storage log causal fp16 Bool.true).property.symm.trans
        (candidate_runs storage log causal fp16))

end
#axiomsClean candidate_runs
#axiomsClean variant_state
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution

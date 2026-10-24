import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardFP
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedValues
import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedBKValues

/-! Transport the two staged, actual fp16 execution witnesses through the
complete-source structural theorem. No new execution success is assumed.
-/
namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section

theorem fp16_log_causal_qk_candidate_runs :
    run (engine model)
      (Typed.narrowCandidate "Y" "X" "M" 11 7 17 5 13 3 4
        Bool.true (some "qk") Bool.true Bool.true).surfaceBody initial =
      some fp16_log_causal_qk_run.val := by
  exact (Floating.forward_run_eq .narrow "Y" "X" "M" 11 7 17 5 13 3 4
    Bool.true (some "qk") Bool.true Bool.true model initial).symm.trans
      fp16_log_causal_qk_run.property

theorem fp16_log_causal_bk_candidate_runs :
    run (engine model)
      (Typed.narrowCandidate "Y" "X" "M" 11 7 17 5 13 3 4
        Bool.true (some "bk") Bool.true Bool.true).surfaceBody initial =
      some fp16_log_causal_bk_run.val := by
  exact (Floating.forward_run_eq .narrow "Y" "X" "M" 11 7 17 5 13 3 4
    Bool.true (some "bk") Bool.true Bool.true model initial).symm.trans
      fp16_log_causal_bk_run.property

/-- A concrete final state for both complete qk programs, with two distinct
finite outputs, a written bottom future lane and an unchanged masked tail. -/
theorem fp16_log_causal_qk_both_values :
    ∃ final : State (WithBot ℝ),
      (∀ commuted : Bool,
        run (engine model)
          (Typed.checkedKernel .narrow commuted "Y" "X" "M"
            11 7 17 5 13 3 4 Bool.true (some "qk") Bool.true Bool.true).surfaceBody
          initial = some final) ∧
      final.mem "Y" 29 = Cell.mk .fp16
        (some (-1 - Real.log (Real.exp (-1) + 1))) ∧
      final.mem "Y" 30 = Cell.mk .fp16
        (some (-Real.log (Real.exp (-1) + 1))) ∧
      final.mem "Y" 31 = Cell.mk .fp16 (none : WithBot ℝ) ∧
      final.mem "Y" 32 = Cell.mk .fp16 (some (7 : ℝ)) := by
  refine ⟨fp16_log_causal_qk_run.val, ?_, fp16_log_causal_qk_values⟩
  intro commuted
  cases commuted
  · exact fp16_log_causal_qk_run.property
  · exact fp16_log_causal_qk_candidate_runs

/-- The bk programs share the actual bk witness, whose active mask values
produce -2 rather than qk's -1 in the shifted first lane. -/
theorem fp16_log_causal_bk_both_values :
    ∃ final : State (WithBot ℝ),
      (∀ commuted : Bool,
        run (engine model)
          (Typed.checkedKernel .narrow commuted "Y" "X" "M"
            11 7 17 5 13 3 4 Bool.true (some "bk") Bool.true Bool.true).surfaceBody
          initial = some final) ∧
      final.mem "Y" 29 = Cell.mk .fp16
        (some (-2 - Real.log (Real.exp (-2) + 1))) ∧
      final.mem "Y" 30 = Cell.mk .fp16
        (some (-Real.log (Real.exp (-2) + 1))) ∧
      final.mem "Y" 31 = Cell.mk .fp16 (none : WithBot ℝ) ∧
      final.mem "Y" 32 = Cell.mk .fp16 (some (7 : ℝ)) := by
  refine ⟨fp16_log_causal_bk_run.val, ?_, fp16_log_causal_bk_values⟩
  intro commuted
  cases commuted
  · exact fp16_log_causal_bk_run.property
  · exact fp16_log_causal_bk_candidate_runs

end
#axiomsClean fp16_log_causal_qk_candidate_runs
#axiomsClean fp16_log_causal_bk_candidate_runs
#axiomsClean fp16_log_causal_qk_both_values
#axiomsClean fp16_log_causal_bk_both_values
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution

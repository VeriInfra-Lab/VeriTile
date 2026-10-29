import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedVariant
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedValueProbe
import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardOrdinaryValueProbe

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Actual cells of the complete typed LOG/CAUSAL fp16 execution. The future
lane receives -4 while the masked tail retains seven, in both source variants. -/
theorem log_causal_cells (v : Bool) :
    (source_run .fp16 Bool.true Bool.true Bool.true v).val.mem "D" 29 =
        Cell.mk .fp16 (some (-2 : ℝ)) ∧
    (source_run .fp16 Bool.true Bool.true Bool.true v).val.mem "D" 31 =
        Cell.mk .fp16 (some (-4 : ℝ)) ∧
    (source_run .fp16 Bool.true Bool.true Bool.true v).val.mem "D" 32 =
        Cell.mk .fp16 (some (7 : ℝ)) := by
  rw [variant_state .fp16 Bool.true Bool.true Bool.true v]
  have hstate : (source_run .fp16 Bool.true Bool.true Bool.true Bool.false).val =
      fp16_log_causal_run.val := Option.some.inj
    ((source_run .fp16 Bool.true Bool.true Bool.true Bool.false).property.symm.trans
      fp16_log_causal_run.property)
  rw [hstate]
  exact fp16_log_causal_values

/-- The ordinary causal branch writes one quarter on a read-active lane
and zero at the future lane. These are interpreter outputs, not inputs. -/
theorem ordinary_causal_cells (v : Bool) :
    (source_run .fp16 Bool.false Bool.true Bool.true v).val.mem "D" 29 =
        Cell.mk .fp16 (some ((1 : ℝ) / 4)) ∧
    (source_run .fp16 Bool.false Bool.true Bool.true v).val.mem "D" 31 =
        Cell.mk .fp16 (some (0 : ℝ)) ∧
    (source_run .fp16 Bool.false Bool.true Bool.true v).val.mem "D" 32 =
        Cell.mk .fp16 (some (7 : ℝ)) := by
  rw [variant_state .fp16 Bool.false Bool.true Bool.true v]
  have hstate : (source_run .fp16 Bool.false Bool.true Bool.true Bool.false).val =
      fp16_ordinary_causal_run.val := Option.some.inj
    ((source_run .fp16 Bool.false Bool.true Bool.true Bool.false).property.symm.trans
      fp16_ordinary_causal_run.property)
  rw [hstate]
  exact fp16_ordinary_causal_values

end
#axiomsClean log_causal_cells
#axiomsClean ordinary_causal_cells
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution

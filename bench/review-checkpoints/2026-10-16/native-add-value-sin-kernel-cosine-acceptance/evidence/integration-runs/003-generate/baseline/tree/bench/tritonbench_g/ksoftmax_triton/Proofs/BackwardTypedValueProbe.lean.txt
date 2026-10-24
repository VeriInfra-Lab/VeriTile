import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedProbe

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 800000
set_option maxRecDepth 16000

/-- Concrete outputs of the independently checked single-profile run. This
includes a negative future-lane value and preservation of the masked tail. -/
theorem fp16_log_causal_values :
    fp16_log_causal_run.val.mem "D" 29 = Cell.mk .fp16 (some (-2 : ℝ)) ∧
    fp16_log_causal_run.val.mem "D" 31 = Cell.mk .fp16 (some (-4 : ℝ)) ∧
    fp16_log_causal_run.val.mem "D" 32 = Cell.mk .fp16 (some (7 : ℝ)) := by
  norm_num [fp16_log_causal_run, initial, model, channel,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex, Fin.sum_univ_succ,
    WithBot.realExp, WithBot.realSub, WithBot.realMul, WithBot.realAdd]
  let total : WithBot ℝ := @Finset.sum (Fin 4) (WithBot ℝ) _ Finset.univ
      (fun x => if 1 < x.val then some 0
        else if x.val < 3 ∧ x.val ≤ 1 then some 2 else some 0)
  change Cell.mk .fp16 (Option.map (fun b : ℝ => 2 - b) total) =
      Cell.mk .fp16 (some (-2 : ℝ)) ∧
    Cell.mk .fp16 (Option.map (fun b : ℝ => -b) total) =
      Cell.mk .fp16 (some (-4 : ℝ)) ∧
    Cell.mk .fp16 (some (if ("D" : String) = "G" then (2 : ℝ)
      else if ("D" : String) = "O" then 0 else 7)) =
      Cell.mk .fp16 (some (7 : ℝ))
  have hsum : total = some (4 : ℝ) := by
    dsimp only [total]
    rw [Fin.sum_univ_four]
    change (some ((((2 : ℝ) + 2) + 0) + 0) : WithBot ℝ) = some (4 : ℝ)
    norm_num
  have hDG : ("D" : String) ≠ "G" := by decide
  have hDO : ("D" : String) ≠ "O" := by decide
  norm_num [hsum, hDG, hDO]

end
#axiomsClean fp16_log_causal_values
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution

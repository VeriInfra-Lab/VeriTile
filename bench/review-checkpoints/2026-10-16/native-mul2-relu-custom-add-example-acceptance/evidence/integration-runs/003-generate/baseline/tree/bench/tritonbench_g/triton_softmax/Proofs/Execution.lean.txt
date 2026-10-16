import bench.tritonbench_g.triton_softmax.Proofs.ExecutionBackend

namespace VeriTile.Bench.TritonBenchG.TritonSoftmax.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

private theorem padded_max (h : (Finset.univ : Finset (Fin 4)).Nonempty) :
    Finset.univ.sup' (α := WithBot ℝ) h (fun x : Fin 4 =>
      if x.val ≤ 1 then some (0 : ℝ) else none) = (some (0 : ℝ) : WithBot ℝ) := by
  apply le_antisymm
  · apply Finset.sup'_le
    intro b hb
    split_ifs
    · exact le_rfl
    · exact bot_le
  · simpa using Finset.le_sup' (α := WithBot ℝ)
      (fun x : Fin 4 => if x.val ≤ 1 then some (0 : ℝ) else none)
      (Finset.mem_univ (0 : Fin 4))

private def witnessRow (x : Fin 4) : WithBot ℝ :=
  if x.val ≤ 1 then some 0 else none

private def witnessMax : WithBot ℝ :=
  Finset.univ.sup' (α := WithBot ℝ) (by exact ⟨0, Finset.mem_univ _⟩) witnessRow

/-- The typed complete fp16 source actually writes one half in both active
cells and retains the masked output sentinel. Both candidate executions are checked. -/
theorem half_output_cells (v : Bool) :
    (source_run .fp16 .fp16 v).val.mem "Y" 7 = Cell.mk .fp16 (some ((1 : ℝ) / 2)) ∧
    (source_run .fp16 .fp16 v).val.mem "Y" 8 = Cell.mk .fp16 (some ((1 : ℝ) / 2)) ∧
    (source_run .fp16 .fp16 v).val.mem "Y" 9 = Cell.mk .fp16 (some 7) := by
  cases v
  all_goals simp [source_run, initial, model, Precision.channel, WithBot.realExp,
    WithBot.realSub, WithBot.realDiv, WithBot.realAdd,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, Real.exp_zero]
  all_goals dsimp only [TileCarrier, FP.Structural.Value, TileShape.axisDim,
    TileShape.insertAxisIndex]
  all_goals change (
    @Cell.mk (WithBot ℝ) .fp16
      (Option.map₂ (fun x y : ℝ => x / y)
        (WithBot.realExp (Option.map (fun x : ℝ => -x) witnessMax))
        (@Finset.sum (Fin 4) (WithBot ℝ) _ Finset.univ
          (fun x => WithBot.realExp (WithBot.realSub (witnessRow x) witnessMax)))) =
        Cell.mk .fp16 (some ((2 : ℝ)⁻¹)) ∧
      Cell.mk .fp16 (some (7 : ℝ)) = Cell.mk .fp16 (some (7 : ℝ)))
  all_goals norm_num [witnessMax, witnessRow, padded_max, Fin.sum_univ_succ,
    WithBot.realExp, WithBot.realSub]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, ← WithBot.coe_nsmul]
  all_goals change (@Cell.mk (WithBot ℝ) .fp16 (some ((2 : ℝ)⁻¹)) =
    Cell.mk .fp16 (some ((1 : ℝ) / 2)))
  all_goals simp only [one_div]
  all_goals rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.TritonSoftmax.NumericExecution

import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardOrdinaryProbe

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 800000
set_option maxRecDepth 16000

private theorem fold_writes_mem {α ι : Type} (xs : List ι)
    (active : ι → Prop) [DecidablePred active]
    (region : ι → RegionName) (offset : ι → Nat) (value : ι → Cell α)
    (s : State α) :
    (xs.foldl (fun t i => if active i then t.write (region i) (offset i) (value i)
      else t) s).mem =
    xs.foldl (fun mem i => if active i then
      fun r o => if r = region i ∧ o = offset i then value i else mem r o
      else mem) s.mem := by
  induction xs generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases hi : active i
    · simp only [hi, ↓reduceIte]
      rfl
    · simp only [hi, ↓reduceIte]

/-- Project the actual interpreter-produced scatter to memory before
normalizing its values. The discarded register record is not part of a memory
observation and must not be repeatedly reduced by arithmetic simplification. -/
private def ordinary_memory :
    {mem : RegionName → Nat → Cell (WithBot ℝ) //
      fp16_ordinary_causal_run.val.mem = mem} := by
  refine ⟨?_, ?_⟩
  swap
  exact fold_writes_mem (TileShape.allIndices [4])
    (fun i => i.1.val < 3) (fun _ => "D") (fun i => 29 + i.1.val) _ _

private def ordinary_total : WithBot ℝ :=
  @Finset.sum (Fin 4) (WithBot ℝ) _ Finset.univ (fun x =>
    Option.map₂ (fun a b : ℝ => a * b)
      (if 1 < x.val then some (0 : ℝ)
        else if x.val < 3 ∧ x.val ≤ 1 then some (2 : ℝ) else some (0 : ℝ))
      (if 1 < x.val then some (0 : ℝ)
        else if x.val < 3 ∧ x.val ≤ 1 then some ((1 : ℝ) / 4) else some (0 : ℝ)))

private theorem ordinary_total_eq : ordinary_total = some (1 : ℝ) := by
  dsimp only [ordinary_total]
  rw [Fin.sum_univ_four]
  change (some (((((2 : ℝ) * (1 / 4)) + 2 * (1 / 4)) + 0 * 0) + 0 * 0) :
    WithBot ℝ) = some (1 : ℝ)
  norm_num

-- Each memory observation has its own opaque proof boundary. The state
-- parameter keeps statement elaboration independent of the concrete run term.
private theorem ordinary_active (t : State (WithBot ℝ))
    (ht : t = fp16_ordinary_causal_run.val) :
    t.mem "D" 29 = (Cell.mk .fp16 (some ((1 : ℝ) / 4)) : Cell (WithBot ℝ)) := by
  subst t
  rw [ordinary_memory.property]
  norm_num [ordinary_memory, initial, model, channel,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex,
    WithBot.realExp, WithBot.realSub, WithBot.realMul, WithBot.realAdd]
  norm_num [Fin.sum_univ_four, ← WithBot.coe_add, WithBot.some_eq_coe,
    WithBot.realSub, WithBot.realMul]
  change Option.map (fun b : ℝ => (1 / 4) * (2 - b)) ordinary_total = some ((1 : ℝ) / 4)
  rw [ordinary_total_eq]
  norm_num

private theorem ordinary_future (t : State (WithBot ℝ))
    (ht : t = fp16_ordinary_causal_run.val) :
    t.mem "D" 31 = (Cell.mk .fp16 (some (0 : ℝ)) : Cell (WithBot ℝ)) := by
  subst t
  rw [ordinary_memory.property]
  norm_num [ordinary_memory, initial, model, channel,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim,
    TileShape.eraseAxis, TileShape.insertAxisIndex,
    WithBot.realExp, WithBot.realSub, WithBot.realMul, WithBot.realAdd]
  norm_num [Fin.sum_univ_four, ← WithBot.coe_add, WithBot.some_eq_coe,
    WithBot.realSub, WithBot.realMul]
  change Option.map (fun _ : ℝ => (0 : ℝ)) ordinary_total = some (0 : ℝ)
  rw [ordinary_total_eq]
  rfl

private theorem ordinary_tail (t : State (WithBot ℝ))
    (ht : t = fp16_ordinary_causal_run.val) :
    t.mem "D" 32 = (Cell.mk .fp16 (some (7 : ℝ)) : Cell (WithBot ℝ)) := by
  subst t
  have hDG : ("D" : String) ≠ "G" := by decide
  have hDO : ("D" : String) ≠ "O" := by decide
  rw [ordinary_memory.property]
  norm_num [ordinary_memory, initial, model, channel,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ,
    ofFloat, toFloat, hDG, hDO]

/-- Independently checked actual output cells, including the unwritten tail. -/
theorem fp16_ordinary_causal_values :
    fp16_ordinary_causal_run.val.mem "D" 29 =
      (Cell.mk .fp16 (some ((1 : ℝ) / 4)) : Cell (WithBot ℝ)) ∧
    fp16_ordinary_causal_run.val.mem "D" 31 =
      (Cell.mk .fp16 (some (0 : ℝ)) : Cell (WithBot ℝ)) ∧
    fp16_ordinary_causal_run.val.mem "D" 32 =
      (Cell.mk .fp16 (some (7 : ℝ)) : Cell (WithBot ℝ)) :=
  ⟨ordinary_active _ rfl, ordinary_future _ rfl, ordinary_tail _ rfl⟩

end
#axiomsClean fp16_ordinary_causal_values
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution

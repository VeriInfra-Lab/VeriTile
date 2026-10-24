import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedProbe

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution
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

private theorem sup_four (f : Fin 4 → WithBot ℝ) :
    Finset.univ.sup f = ((f 0 ⊔ f 1) ⊔ f 2) ⊔ f 3 := by
  have hFin : (Finset.univ : Finset (Fin 4)) = {0, 1, 2, 3} := by decide
  rw [hFin]
  simp [sup_assoc]

/-- Observe the actual final scatter without repeatedly normalizing its
intermediate register record. The source program itself has already run. -/
private def qk_memory :
    {mem : RegionName → Nat → Cell (WithBot ℝ) //
      fp16_log_causal_qk_run.val.mem = mem} := by
  refine ⟨?_, ?_⟩
  swap
  exact fold_writes_mem (TileShape.allIndices [4])
    (fun i => i.1.val < 3) (fun _ => "Y") (fun i => 29 + i.1.val) _ _

private theorem output_cells (t : State (WithBot ℝ))
    (ht : t = fp16_log_causal_qk_run.val) :
    t.mem "Y" 29 = (Cell.mk .fp16 (some (-1 - Real.log (Real.exp (-1) + 1))) : Cell (WithBot ℝ)) ∧
    t.mem "Y" 30 = (Cell.mk .fp16 (some (-Real.log (Real.exp (-1) + 1))) : Cell (WithBot ℝ)) ∧
    t.mem "Y" 31 = (Cell.mk .fp16 (none : WithBot ℝ) : Cell (WithBot ℝ)) ∧
    t.mem "Y" 32 = (Cell.mk .fp16 (some (7 : ℝ)) : Cell (WithBot ℝ)) := by
  subst t
  rw [qk_memory.property]
  have hMX : ("M" : String) ≠ "X" := by decide
  have hYX : ("Y" : String) ≠ "X" := by decide
  have hYM : ("Y" : String) ≠ "M" := by decide
  norm_num [qk_memory, initial, model, hMX, hYX, hYM,
    FP.Structural.store, State.write, TileShape.allIndices, List.finRange_succ, bop,
    Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
    Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex,
    WithBot.realExp, WithBot.realLog, WithBot.realSub, WithBot.realAdd,
    Finset.sup'_eq_sup, sup_four, Fin.sum_univ_four,
    ← WithBot.coe_add, WithBot.some_eq_coe]
  -- Fix the reduction index to Fin 4 before rewriting the finite lattice
  -- and sum; simplification alone leaves their dependent axis type opaque.
  let row : Fin 4 → WithBot ℝ := fun i => WithBot.realAdd
    (if 1 < i.val then none else if i.val < 3 ∧ i.val ≤ 1 then 0 else none)
    (if i.val < 3 ∧ i.val ≤ 1 then
      some (if i = 0 then 5 else if 13 + i.val = 14 then 6
        else if 13 + i.val = 26 then 26 else if 13 + i.val = 27 then 28 else 0)
      else none)
  let maximum : WithBot ℝ := Finset.univ.sup row
  let total : WithBot ℝ := ∑ i : Fin 4,
    WithBot.realExp (WithBot.realSub (row i) maximum)
  change WithBot.realSub
      (WithBot.realSub (WithBot.realAdd (some 0) (some 5)) maximum)
      (WithBot.realLog total) = some (-1 - Real.log (Real.exp (-1) + 1)) ∧
    WithBot.realSub
      (WithBot.realSub (WithBot.realAdd (some 0) (some 6)) maximum)
      (WithBot.realLog total) = some (-Real.log (Real.exp (-1) + 1))
  have hrow0 : row 0 = some 5 := by
    change some ((0 : ℝ) + 5) = some 5
    norm_num
  have hrow1 : row 1 = some 6 := by
    change some ((0 : ℝ) + 6) = some 6
    norm_num
  have hrow2 : row 2 = none := rfl
  have hrow3 : row 3 = none := rfl
  have hmaximum : maximum = some 6 := by
    dsimp only [maximum]
    rw [sup_four, hrow0, hrow1, hrow2, hrow3]
    change max (max (max ((5 : ℝ) : WithBot ℝ) ((6 : ℝ) : WithBot ℝ)) ⊥) ⊥ =
      ((6 : ℝ) : WithBot ℝ)
    rw [max_bot_right, max_bot_right, max_eq_right]
    exact_mod_cast (show (5 : ℝ) ≤ 6 by norm_num)
  have htotal : total = some (Real.exp (-1) + 1) := by
    dsimp only [total]
    rw [Fin.sum_univ_four, hmaximum, hrow0, hrow1, hrow2, hrow3]
    change (some (((Real.exp ((5 : ℝ) - 6) + Real.exp (6 - 6)) + 0) + 0) :
      WithBot ℝ) = some (Real.exp (-1) + 1)
    norm_num
  norm_num [hmaximum, htotal, WithBot.realAdd, WithBot.realSub, WithBot.realLog]

/-- Complete original-source outputs, including the extended negative-infinity
payload at a written future lane and the unchanged masked tail. -/
theorem fp16_log_causal_qk_values :
    fp16_log_causal_qk_run.val.mem "Y" 29 =
      (Cell.mk .fp16 (some (-1 - Real.log (Real.exp (-1) + 1))) : Cell (WithBot ℝ)) ∧
    fp16_log_causal_qk_run.val.mem "Y" 30 =
      (Cell.mk .fp16 (some (-Real.log (Real.exp (-1) + 1))) : Cell (WithBot ℝ)) ∧
    fp16_log_causal_qk_run.val.mem "Y" 31 =
      (Cell.mk .fp16 (none : WithBot ℝ) : Cell (WithBot ℝ)) ∧
    fp16_log_causal_qk_run.val.mem "Y" 32 =
      (Cell.mk .fp16 (some (7 : ℝ)) : Cell (WithBot ℝ)) :=
  output_cells _ rfl

end
#axiomsClean fp16_log_causal_qk_values
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution

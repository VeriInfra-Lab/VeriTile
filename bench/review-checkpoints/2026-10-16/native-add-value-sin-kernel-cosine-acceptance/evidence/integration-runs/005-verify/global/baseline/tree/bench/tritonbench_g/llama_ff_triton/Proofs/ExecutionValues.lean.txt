import bench.tritonbench_g.llama_ff_triton.Proofs.Execution

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false

private theorem sum_fin2 {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

private theorem card_fin2 (I : Fintype (Fin 2)) : @Fintype.card (Fin 2) I = 2 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  rfl

/-- Both weight profiles and both complete variants successfully write a
nonzero SwiGLU result. A masked output address and unrelated typed cells keep
their integer sentinels. The byte profile preserves the original weight bytes. -/
theorem numeric_cells (fp8 variant : Bool) :
    (numeric_run fp8 variant).val.mem "Y" 0 = .mk .fp16 (some (Real.sigmoid 1)) ∧
    (numeric_run fp8 variant).val.mem "Y" 2 = .mk .int 37 ∧
    (numeric_run fp8 variant).val.mem "Y" 1 = .mk .int 37 ∧
    (numeric_run fp8 variant).val.mem "W1" 25 =
      (if fp8 then .mk .int 60 else .mk .fp16 (some 1)) := by
  cases fp8 <;> cases variant
  all_goals simp (maxSteps := 200000) only [numeric_run, store_run, address_run,
    math_run, loop_run, iteration0_run, iteration1_run, iteration2_run,
    iteration3_run, iteration4_run, prefix_run,
    FP.Structural.store, State.write, State.setReg_mem,
    Region.cast, TileShape.allIndices, List.finRange_succ,
    List.foldl_cons, List.foldl_nil, List.flatMap_cons, List.flatMap_nil,
    List.map_cons, List.map_nil, List.cons_append, List.nil_append,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar_data,
    Tile.select, Tile.ptrAdd, ofFloat, toFloat]
  all_goals simp (maxSteps := 200000) [initial, model, State.setReg, Region.cast,
    Function.comp_def, Tile.scalar, -Tile.scalar_eta, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.insertAxisIndex, TileShape.eraseAxis, sum_fin2, Tile.dot,
    WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
    WithBot.realRsqrt, WithBot.realPowi, WithBot.realSigmoid, zpow_ofNat, pow_two,
    Float8E5Bits.decodeRat, Float8E5Bits.ofInt, Float8E5Bits.exponent,
    Float8E5Bits.fraction, Float8E5Bits.sign]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind,
    bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Function.comp_def, List.finRange_succ, List.foldl, List.flatMap, sum_fin2, card_fin2]
  all_goals
    have h : (2 : WithBot ℝ) * ((1 / 2 : ℝ) : WithBot ℝ) = (some 1 : WithBot ℝ) := by
      change ((2 : ℝ) : WithBot ℝ) * ((1 / 2 : ℝ) : WithBot ℝ) = ((1 : ℝ) : WithBot ℝ)
      rw [← WithBot.coe_mul]
      norm_num
    simp only [h]
    simp

theorem output_positive : 0 < Real.sigmoid 1 := Real.sigmoid_pos 1

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution

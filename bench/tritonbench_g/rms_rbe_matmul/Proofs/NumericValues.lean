import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericExecution

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 6000
set_option linter.unusedSimpArgs false

private theorem sum_two {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

private theorem card_two (I : Fintype (Fin 2)) : @Fintype.card (Fin 2) I = 2 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  rfl

/-- Concrete observations of the complete typed execution: actual output
conversion, nonzero batch address, masked/before-output sentinels and preserved
byte inputs. The algebra computes real arithmetic and does not assert IEEE
rounding or a packed output-byte representation. -/
theorem numeric_cells (fp8 variant : Bool) :
    (numeric_run fp8 variant).val.mem "Y" 17 =
      (if fp8 then .mk .f8e5 (some 2) else .mk .fp16 (some 2)) ∧
    (numeric_run fp8 variant).val.mem "Y" 19 = .mk .int 37 ∧
    (numeric_run fp8 variant).val.mem "Y" 0 = .mk .int 37 ∧
    (numeric_run fp8 variant).val.mem "W" 3 =
      (if fp8 then .mk .int 64 else .mk .fp16 (some 2)) ∧
    (numeric_run fp8 variant).val.mem "R" 2 =
      (if fp8 then .mk .int 60 else .mk .fp16 (some 1)) := by
  cases fp8 <;> cases variant
  all_goals
    simp (maxSteps := 1000000) [numeric_run, store_run, address_run, math_run, loop_run,
      prefix_run, iteration0_run, iteration1_run, iteration2_run, iteration3_run, State.setReg, State.write, FP.Structural.store,
      initial, Region.cast, Tile.scalar, -Tile.scalar_eta,
      Function.comp_def, Tile.select, Tile.ptrAdd, TileShape.allIndices, List.finRange_succ,
      List.foldl, bop, Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
      Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.insertAxisIndex,
      TileShape.eraseAxis, sum_two, Tile.dot,
      WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
      WithBot.realRsqrt, WithBot.realPowi, zpow_ofNat, pow_two,
      Float8E5Bits.decodeRat, Float8E5Bits.ofInt, Float8E5Bits.exponent,
      Float8E5Bits.fraction, Float8E5Bits.sign]
    all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind,
      bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
      Function.comp_def, List.finRange_succ, List.foldl, List.flatMap, sum_two, card_two,
      two_nsmul, ← WithBot.coe_mul]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution

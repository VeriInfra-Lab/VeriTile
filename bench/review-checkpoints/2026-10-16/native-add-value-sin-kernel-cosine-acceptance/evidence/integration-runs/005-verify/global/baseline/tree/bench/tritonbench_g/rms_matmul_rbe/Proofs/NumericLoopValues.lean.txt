import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericLoop

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
theorem numeric_sum_two {α : Type*} [AddCommMonoid α]
    (I : Fintype (Fin 2)) (f : Fin 2 → α) :
    @Finset.sum (Fin 2) α _ (@Finset.univ _ I) f = f 0 + f 1 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  simp [Fin.sum_univ_succ]

theorem numeric_card_two (I : Fintype (Fin 2)) : @Fintype.card (Fin 2) I = 2 := by
  have hI : I = Fin.fintype 2 := Subsingleton.elim _ _
  rw [hI]
  rfl


def value : Pass → ℝ | .q => 2 | .k => 3 | .v => 4

theorem loop_registers (fp8 : Bool) (p : Pass) (s : State (WithBot ℝ))
    :
    (loop_run fp8 p s).val.regs .real [1, 2] "x_sum" = some (fun _ => some 1) ∧
    (loop_run fp8 p s).val.regs .real [1, 2] "accumulator" = some (fun _ => some (2 * value p)) ∧
    (loop_run fp8 p s).val.regs .nat [] "pid_m" = some (fun _ => 0) ∧
    (loop_run fp8 p s).val.regs .nat [] "pid_n" = some (fun _ => 0) ∧
    (loop_run fp8 p s).val.regs .nat [] "pid_batch" = some (fun _ => 0) ∧
    (loop_run fp8 p s).val.mem = (startState fp8 p s).mem ∧
    (loop_run fp8 p s).val.pids = (startState fp8 p s).pids := by
  cases fp8 <;> cases p
  all_goals
      dsimp only [loop_run, prefix_run, iteration0_run, iteration1_run, iteration2_run,
        iteration3_run, State.setReg, startState, memory, before, value]
      simp (maxSteps := 2000000) [loop_run, prefix_run, iteration0_run, iteration1_run, iteration2_run, iteration3_run,
        State.setReg, State.write, FP.Structural.store,
        startState, memory, before, value, Region.cast, Tile.scalar, -Tile.scalar_eta,
        Function.comp_def, Tile.select, Tile.ptrAdd, TileShape.allIndices, List.finRange_succ,
        List.foldl, bop, Broadcast.leftIndex, Broadcast.rightIndex, ofFloat, toFloat,
        Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.insertAxisIndex,
        TileShape.eraseAxis, numeric_sum_two, Tile.dot,
        WithBot.realAdd, WithBot.realSub, WithBot.realMul, WithBot.realDiv,
        WithBot.realRsqrt, WithBot.realPowi, zpow_ofNat, pow_two,
        Float8E5Bits.decodeRat, Float8E5Bits.ofInt, Float8E5Bits.exponent,
        Float8E5Bits.fraction, Float8E5Bits.sign]
      all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, Option.map₂, Option.map, Option.bind,
        bop, Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
        Function.comp_def, List.finRange_succ, List.foldl, List.flatMap, numeric_sum_two,
        numeric_card_two, two_nsmul, ← WithBot.coe_mul]
  all_goals repeat' constructor
  all_goals try apply congrArg some
  all_goals funext i
  all_goals norm_num [bop, Broadcast.leftIndex, Broadcast.rightIndex,
    WithBot.realPowi, WithBot.realAdd, Option.map₂, Option.bind, zpow_ofNat]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution

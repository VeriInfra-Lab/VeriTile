import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericLoopValues

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- Compact state after the actual three post-loop arithmetic assignments. -/
def mathState (t : State (WithBot ℝ)) (p : Pass) : State (WithBot ℝ) :=
  ((t.setReg "x_mean" .real [1] (fun _ => some 4)).setReg "x_norm" .real [1]
    (fun _ => some (1 / 2))).setReg "accumulator" .real [1, 2] (fun _ => some (value p))

theorem math_step (fp8 : Bool) (p : Pass) (t : State (WithBot ℝ))
    (hx : t.regs .real [1, 2] "x_sum" = some (fun _ => some 1))
    (ha : t.regs .real [1, 2] "accumulator" = some (fun _ => some (2 * value p))) :
    run (engine model) ((pass fp8 Bool.false p).surfaceBody.drop 13 |>.take 3) t =
      some (mathState t p) := by
  cases fp8 <;> cases p
  all_goals simp (maxSteps := 1000000) [mathState, value, hx, ha, pass, TypedSource.kernel, ComputeKernel.surfaceBody, run, step,
    FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
    Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
    FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
    Broadcast.leftIndex, Broadcast.rightIndex, Tile.scalar, -Tile.scalar_eta,
    Float8E5Bits.isInt8, Float8E5Bits.ofInt,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.eraseAxis, TileShape.insertAxisIndex,
    numeric_sum_two, WithBot.realAdd, WithBot.realMul, WithBot.realDiv, WithBot.realRsqrt]
  all_goals norm_num [WithBot.some_eq_coe, ← WithBot.coe_add, ← WithBot.coe_mul,
    Option.map₂, Option.map, numeric_card_two, numeric_sum_two, two_nsmul]
  all_goals
    congr 1
    · congr 1
      · congr 1
        funext i
        norm_num [bop, Broadcast.leftIndex, Broadcast.rightIndex, Option.bind,
          WithBot.some_eq_coe, ← WithBot.coe_add, ← WithBot.coe_mul]
      · funext i
        norm_num [bop, Broadcast.leftIndex, Broadcast.rightIndex, Option.bind,
          Function.comp_def, WithBot.some_eq_coe, ← WithBot.coe_add, ← WithBot.coe_mul]
    · funext i
      norm_num [bop, Broadcast.leftIndex, Broadcast.rightIndex, Option.bind,
        WithBot.some_eq_coe, ← WithBot.coe_add, ← WithBot.coe_mul]

def math_run (fp8 : Bool) (p : Pass) (s : State (WithBot ℝ))
    :
    {t : State (WithBot ℝ) // run (engine model) ((pass fp8 Bool.false p).surfaceBody.drop 13 |>.take 3)
      (loop_run fp8 p s).val = some t} :=
  ⟨mathState (loop_run fp8 p s).val p,
    math_step fp8 p _ (loop_registers fp8 p s).1 (loop_registers fp8 p s).2.1⟩

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution

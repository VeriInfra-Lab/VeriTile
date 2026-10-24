import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericIteration1

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def iteration2_run (fp8 : Bool) :
    {t : State (WithBot ℝ) // run (engine model) ((loopCode fp8).drop 8 |>.take 4)
      (iteration1_run fp8).val = some t} := by
  cases fp8
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [loopCode, kernelExample, TypedSource.kernel,
      ComputeKernel.surfaceBody, run, step, FP.Structural.range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
      initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
      FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
      Float8E5Bits.isInt8, Float8E5Bits.ofInt, State.setReg, prefix_run, iteration0_run, iteration1_run]
    try rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution

import bench.tritonbench_g.rms_rbe_matmul.Proofs.NumericMath

namespace VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- The actual output addresses and mask, including the batch offset. -/
def address_run (fp8 variant : Bool) :
    {t : State (WithBot ℝ) // run (engine model) (((kernelExample fp8 variant).surfaceBody.drop 16 |>.take 4))
      ((math_run fp8).val) = some t} := by
  cases fp8 <;> cases variant
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [kernelExample, TypedSource.kernel,
      ComputeKernel.surfaceBody, run, step, FP.Structural.range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
      initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
      FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
      Float8E5Bits.isInt8, Float8E5Bits.ofInt, State.setReg, prefix_run, loop_run, iteration0_run, iteration1_run, iteration2_run, iteration3_run, math_run]
    try rfl


end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsRbeMatmul.NumericExecution

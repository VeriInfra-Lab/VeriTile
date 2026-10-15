import bench.tritonbench_g.llama_ff_triton.Proofs.ExecutionIteration2

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 2000000
set_option maxRecDepth 3000
set_option linter.unusedSimpArgs false
def iteration3_run (fp8 : Bool) :
    {t : State (WithBot ℝ) // run (engine model) ((loopCode fp8).drop 12 |>.take 4)
      (iteration2_run fp8).val = some t} := by
  cases fp8
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [loopCode, kernelExample, TypedSource.kernel, TypedSource.original,
      TypedSource.candidate, TypedSource.originalFP8, TypedSource.candidateFP8,
      ComputeKernel.surfaceBody, run, step, FP.Structural.range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
      initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
      FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
      Float8E5Bits.isInt8, Float8E5Bits.ofInt, prefix_run, iteration0_run, iteration1_run, iteration2_run]
    try rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution

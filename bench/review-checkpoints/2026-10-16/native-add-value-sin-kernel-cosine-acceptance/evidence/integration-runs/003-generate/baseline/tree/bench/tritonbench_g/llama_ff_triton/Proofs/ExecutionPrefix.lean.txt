import bench.tritonbench_g.llama_ff_triton.Proofs.ExecutionModel

namespace VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
def prefix_run (fp8 : Bool) :
    {t : State (WithBot ℝ) // run (engine model) ((kernelExample fp8 Bool.false).surfaceBody.take 13)
      (initial fp8) = some t} := by
  cases fp8
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [kernelExample, TypedSource.kernel, TypedSource.original,
      TypedSource.candidate, TypedSource.originalFP8, TypedSource.candidateFP8,
      ComputeKernel.surfaceBody, run, step, FP.Structural.range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
      initial, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
      FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
      Float8E5Bits.isInt8, Float8E5Bits.ofInt]
    try rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LlamaFfTriton.NumericExecution

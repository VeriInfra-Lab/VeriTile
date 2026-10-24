import bench.tritonbench_g.rms_matmul_rbe.Proofs.NumericModel

namespace VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
def prefix_run (fp8 : Bool) (p : Pass) (s : State (WithBot ℝ))
    :
    {t : State (WithBot ℝ) // run (engine model) ((pass fp8 Bool.false p).surfaceBody.take 12)
      (startState fp8 p s) = some t} := by
  cases fp8 <;> cases p
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 1000000) [
      pass, before, TypedSource.kernel, startState, memory,
      ComputeKernel.surfaceBody, run, step, FP.Structural.range,
      FP.Structural.evalExpr, evalComputeOp, evalOp_unfold, evalOffsets, evalDeltas,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, integralFloorDiv, integralMod, bop, natLt,
      Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
      FP.Structural.store, TileShape.allIndices, List.finRange_succ, Tile.select, Tile.ptrAdd,
      Float8E5Bits.isInt8, Float8E5Bits.ofInt]
    try rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsMatmulRbe.NumericExecution

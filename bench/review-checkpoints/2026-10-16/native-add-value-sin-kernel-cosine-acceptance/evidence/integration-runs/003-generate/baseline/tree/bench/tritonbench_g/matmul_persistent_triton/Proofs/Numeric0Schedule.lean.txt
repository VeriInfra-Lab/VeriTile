import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericPrefix

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
def iteration0_schedule_run (storage : TypedSource.Storage) :
    {t : State (WithBot ℝ) // run (engine model)
      ((loopCode storage).drop 0 |>.take 2) ((prefix_run storage).val.setReg "_i" .nat [] (fun _ => 0)) = some t} := by
  cases storage
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 300000) [loopCode, FPProof.loop, SourceChecks.loopBody,
      sample, Parameters.source, TypedSource.kernel,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.range, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, natLt,
      initial, channel, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
      FP.Structural.store, TileShape.allIndices, List.finRange_succ, prefix_run]
    try rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution

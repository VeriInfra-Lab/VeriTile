import bench.tritonbench_g.matmul_persistent_triton.Proofs.NumericModel

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false
def loopCode (storage : TypedSource.Storage) : List ComputeStmt :=
  FPProof.loop sample storage Bool.false

def prefix_run (storage : TypedSource.Storage) :
    {t : State (WithBot ℝ) // run (engine model)
      ((sample.source storage Bool.false).surfaceBody.take 16) (initial storage) = some t} := by
  cases storage
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 300000) [sample, Parameters.source, TypedSource.kernel,
      ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
      run, step, FP.Structural.range, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      evalOffsets, evalDeltas, Option.bind, engine, Algebra.withDefaultPrecision,
      resolvePrecision, model, ComputeDType.eraseDType, numeric, bop, natLt,
      initial, channel, Cell.read, ofFloat, toFloat, TileShape.axisDim, Region.cast,
      FP.Structural.store, TileShape.allIndices, List.finRange_succ]
    try rfl

end
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.NumericExecution

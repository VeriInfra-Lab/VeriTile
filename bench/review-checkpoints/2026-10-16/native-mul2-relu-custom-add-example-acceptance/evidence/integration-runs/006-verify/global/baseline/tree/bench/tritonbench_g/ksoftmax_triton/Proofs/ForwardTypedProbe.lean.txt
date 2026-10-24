import bench.tritonbench_g.ksoftmax_triton.Proofs.ForwardTypedModel

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 800000
set_option maxRecDepth 16000

/-- Complete original source run: fp16 X/M/Y, LOG, CAUSAL, IS_FP16, qk.
The witness is produced by reducing the actual interpreter, including the
typed maximum, cast, exp, sum, log and masked store. -/
def fp16_log_causal_qk_run :
    {t : State (WithBot ℝ) // run (engine model)
      (Typed.narrowOriginal "Y" "X" "M" 11 7 17 5 13 3 4
        Bool.true (some "qk") Bool.true Bool.true).surfaceBody initial = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (maxSteps := 100000) [Typed.narrowOriginal, ComputeKernel.surfaceBody,
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, bop, natLt, natLe, initial, Cell.read,
    ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg, FP.Structural.store,
    Tile.reduceMax, Tile.reduceMaxDrop]
  try rfl

end
#axiomsClean fp16_log_causal_qk_run
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Forward.TypedExecution

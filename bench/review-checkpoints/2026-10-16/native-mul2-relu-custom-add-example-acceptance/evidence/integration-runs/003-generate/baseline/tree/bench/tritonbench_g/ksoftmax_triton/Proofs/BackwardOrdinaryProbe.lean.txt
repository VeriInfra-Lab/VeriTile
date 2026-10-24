import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedModel

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 800000
set_option maxRecDepth 16000

/-- A separate complete ordinary causal source run, retaining the actual
loads, typed reduction, final cast and masked store. -/
def fp16_ordinary_causal_run :
    {t : State (WithBot ℝ) // run (engine model)
      (Typed.homogeneousKernel .fp16 Bool.false "D" "G" "O" 11 7 13 5 17 3 3 4
        Bool.false Bool.true Bool.true).surfaceBody (initial .fp16 Bool.false) = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (maxSteps := 100000) [Typed.homogeneousKernel, ComputeKernel.surfaceBody,
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ComputeDType.eraseDType, numeric, bop, natLt, natLe, initial, channel, Cell.read,
    ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg, FP.Structural.store]
  try rfl

end
#axiomsClean fp16_ordinary_causal_run
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution

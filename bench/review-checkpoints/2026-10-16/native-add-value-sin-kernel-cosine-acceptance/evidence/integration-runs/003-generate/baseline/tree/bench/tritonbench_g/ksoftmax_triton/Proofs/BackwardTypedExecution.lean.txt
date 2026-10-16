import bench.tritonbench_g.ksoftmax_triton.Proofs.BackwardTypedModel

namespace VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
open scoped Classical
set_option maxHeartbeats 8000000
set_option maxRecDepth 16000

/-- Successful full-source execution in all six homogeneous storage
profiles, eight flag profiles and both variants: 96 independent paths.
Both program IDs are nonzero, D=4/K=3 leaves an unwritten tail, and the three
stride pairs differ. Registers start empty, so stale register dtype lookups
cannot be hidden by preloaded intermediates. -/
def source_run (storage : Typed.Storage) (log causal fp16 v : Bool) :
    {t : State (WithBot ℝ) // run (engine model)
      (Typed.homogeneousKernel storage v "D" "G" "O" 11 7 13 5 17 3 3 4
        log causal fp16).surfaceBody (initial storage log) = some t} := by
  cases storage <;> cases log <;> cases causal <;> cases fp16 <;> cases v
  all_goals
    refine ⟨?_, ?_⟩
    swap
    simp (maxSteps := 500000) [Typed.homogeneousKernel, ComputeKernel.surfaceBody,
      run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      Option.bind, engine, Algebra.withDefaultPrecision, resolvePrecision, model,
      ComputeDType.eraseDType, numeric, bop, natLt, natLe, initial, channel, Cell.read,
      ofFloat, toFloat, TileShape.axisDim, Region.cast, State.setReg, FP.Structural.store]
    try rfl

end
#axiomsClean source_run
end VeriTile.Bench.TritonBenchG.KsoftmaxTriton.Backward.TypedExecution

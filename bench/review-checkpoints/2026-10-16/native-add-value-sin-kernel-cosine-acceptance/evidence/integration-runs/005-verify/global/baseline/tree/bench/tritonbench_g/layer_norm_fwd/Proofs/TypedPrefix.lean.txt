import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedModel

/-! Complete typed execution of the real source's pointer and input-load
prefix. It starts with empty registers and derives the control facts consumed
by later phases; the inactive lanes keep the source's explicit zero fallback. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def stagePrefix (rms : Bool) (p : ℝ) : List ComputeStmt :=
  (original (flags rms) regions strides 2 2 1 p 4 legal_block).surfaceBody.take 10

theorem prefix_success (rms : Bool) (p : ℝ) :
    ∃ s, run (engine model) (stagePrefix rms p) initial = some s ∧ Ready s := by
  simp [stagePrefix, original, flags, regions, strides, ComputeKernel.surfaceBody,
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, ComputeDType.eraseDType, numeric, integralMod, bop, natLt,
    numericLt, initial, Cell.read, ofFloat, toFloat, Region.cast,
    Random.i32ArangeSize, Ready]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution

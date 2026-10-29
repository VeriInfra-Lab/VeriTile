import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedNormalization

/-! All remaining typed source statements, including unspecified masked
coefficient lanes, both affine computations and both output stores. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def affine (rms : Bool) (p : ℝ) : List ComputeStmt :=
  (original (flags rms) regions strides 2 2 1 p 4 legal_block).surfaceBody.drop 18

theorem affine_success (rms : Bool) (p : ℝ) (s : State (WithBot ℝ))
    (hs : NormalizedReady rms s) :
    ∃ t, run (engine model) (affine rms p) s = some t := by
  rcases hs with ⟨⟨hrow, hcols, hy, hx1, hres, hresout, hy1, x, hx⟩, ⟨std, hstd⟩, hmean⟩
  cases rms
  · obtain ⟨mean, hm⟩ := hmean rfl
    simp [affine, original, flags, regions, strides, ComputeKernel.surfaceBody,
      run, step, store, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
      numericLt, Region.cast, Random.i32ArangeSize, hcols, hy, hy1, hx, hstd, hm]
  · simp [affine, original, flags, regions, strides, ComputeKernel.surfaceBody,
      run, step, store, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
      numericLt, Region.cast, Random.i32ArangeSize, hcols, hy, hy1, hx, hstd]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution

import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedSecondStream

/-! Typed normalization executes both retained branches and scalar stores.
Its values remain outputs of the actual algebra operations; no mean, variance
or reciprocal standard deviation is supplied to the source. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def normalization (rms : Bool) (p : ℝ) : List ComputeStmt :=
  ((original (flags rms) regions strides 2 2 1 p 4 legal_block).surfaceBody.drop 15).take 3

def NormalizedReady (rms : Bool) (s : State (WithBot ℝ)) : Prop :=
  Ready s ∧ (∃ std, s.regs .real [] "rstd" = some std) ∧
    (rms = Bool.false → ∃ mean, s.regs .real [] "mean" = some mean)

theorem normalization_success (rms : Bool) (p : ℝ) (s : State (WithBot ℝ)) (hs : Ready s) :
    ∃ t, run (engine model) (normalization rms p) s = some t ∧ NormalizedReady rms t := by
  rcases hs with ⟨hrow, hcols, hy, hx1, hres, hresout, hy1, x, hx⟩
  cases rms
  all_goals
    simp [normalization, original, flags, regions, strides, ComputeKernel.surfaceBody,
      run, step, store, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
      Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
      numericLt, Region.cast, Random.i32ArangeSize, TileShape.axisDim,
      hrow, hcols, hy, hx1, hres, hresout, hy1, hx, NormalizedReady, Ready]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution

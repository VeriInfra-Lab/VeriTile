import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedPrefix

/-! Actual first rowscale and dropout, including its boolean masked store.
The proof needs only established source registers. All loaded integer seeds
are accepted after the actual modulo-2^32 cast; no RNG value is supplied. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def firstStream (rms : Bool) (p : ℝ) : List ComputeStmt :=
  ((original (flags rms) regions strides 2 2 1 p 4 legal_block).surfaceBody.drop 10).take 2

theorem firstStream_success (rms : Bool) (p : ℝ) (s : State (WithBot ℝ)) (hs : Ready s) :
    ∃ t, run (engine model) (firstStream rms p) s = some t ∧ Ready t := by
  rcases hs with ⟨hrow, hcols, hy, hx1, hres, hresout, hy1, x, hx⟩
  have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
    (seed_valid_any ((s.mem ("SEEDS" : RegionName) 1).read .int))
  have ho : ∀ j : Fin 4, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
    fun j => hv.2 (j, PUnit.unit)
  have hu := hv.1
  simp at hu
  simp [firstStream, original, flags, regions, strides, ComputeKernel.surfaceBody,
    run, step, store, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, ComputeDType.eraseDType, numeric, integralMod, bop, natLt,
    numericLt, compare_available, seed_valid_any, Region.cast, Random.i32ArangeSize,
    hrow, hcols, hy, hx1, hres, hresout, hy1, hx, hu, ho, Ready]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution

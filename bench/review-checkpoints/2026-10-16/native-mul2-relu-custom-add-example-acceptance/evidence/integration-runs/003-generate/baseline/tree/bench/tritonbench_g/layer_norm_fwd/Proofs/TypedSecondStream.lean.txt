import bench.tritonbench_g.layer_norm_fwd.Proofs.TypedFirstStream

/-! The complete typed second stream, residual addition and ResidualOut store.
The second seed and rowscale use M + row = 3, and its mask starts at offset 6. -/
namespace VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

def secondStream (rms : Bool) (p : ℝ) : List ComputeStmt :=
  ((original (flags rms) regions strides 2 2 1 p 4 legal_block).surfaceBody.drop 12).take 3

theorem secondStream_success (rms : Bool) (p : ℝ) (s : State (WithBot ℝ)) (hs : Ready s) :
    ∃ t, run (engine model) (secondStream rms p) s = some t ∧ Ready t := by
  rcases hs with ⟨hrow, hcols, hy, hx1, hres, hresout, hy1, x, hx⟩
  have hv := (Random.SourceProfile.inputsValid_iff _ _ _).mp
    (seed_valid_any ((s.mem ("SEEDS" : RegionName) 3).read .int))
  have ho : ∀ j : Fin 4, Random.IntegerType.i32.inRange (j.val : Int) = Bool.true :=
    fun j => hv.2 (j, PUnit.unit)
  have hu := hv.1
  simp at hu
  simp [secondStream, original, flags, regions, strides, ComputeKernel.surfaceBody,
    run, step, store, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, ComputeDType.eraseDType, numeric, integralMod, bop, natLt,
    numericLt, compare_available, seed_valid_any, Region.cast, Random.i32ArangeSize,
    hrow, hcols, hy, hx1, hres, hresout, hy1, hx, hu, ho, Ready]

end
end VeriTile.Bench.TritonBenchG.LayerNormFwd.TypedExecution

import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedModel
import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedProgram

/-! Availability of source registers, without assumptions on precomputed
input tiles, carries or output values. Initialization runs from arbitrary state. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def Ready (u : State (WithBot ℝ)) : Prop :=
  (∃ v, u.regs .nat [] "i_bh" = some v) ∧
  (∃ v, u.regs .real [] "b_z" = some v)

theorem prefix_success (dtype out_dtype : FloatDType) (T BT : Nat)
    (s o : RegionName) (u : State (WithBot ℝ)) :
    ∃ v, run (engine model)
      ((TypedSource.source Bool.false dtype out_dtype s o T BT).surfaceBody.take 2) u =
        some v ∧ Ready v := by
  simp [TypedSource.source, TypedSource.original, ComputeKernel.surfaceBody,
    run, step, FP.Structural.evalExpr, evalComputeOp, evalOp_unfold,
    Option.bind, ComputeDType.eraseDType, numeric, bop, Region.cast, Ready]

theorem mirror_ready (u : State (WithBot ℝ)) (j : Int) (h : Ready u) :
    Ready (u.setReg "__rev_i_t" .int [] (fun _ => j)) := by
  rcases h with ⟨⟨a,ha⟩,⟨z,hz⟩⟩
  refine ⟨⟨a,?_⟩,⟨z,?_⟩⟩ <;> simp [ha,hz]

#axiomsClean prefix_success
#axiomsClean mirror_ready
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution

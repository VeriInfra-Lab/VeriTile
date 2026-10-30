import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.TypedReady

/-! Every statement of the actual typed loop succeeds in the consistency
backend, including the masked load, real reduction and inclusive scan,
carry update, output conversion and masked store. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem body_success (T BT : Nat) (s o : RegionName)
    (u : State (WithBot ℝ)) (j : Int) (hj : j < 1) (h : Ready u) :
    ∃ v, run (engine model) (TypedProgram.parts Bool.false .bf16 .fp16 s o T BT).body
      (u.setReg "__rev_i_t" .int [] (fun _ => j)) = some v ∧ Ready v := by
  rcases h with ⟨⟨a,ha⟩,⟨z,hz⟩⟩
  have hnonneg : 0 ≤ -j := by omega
  have hj0 : j ≤ 0 := by omega
  simp [TypedProgram.parts, TypedSource.source, TypedSource.original,
    ComputeKernel.surfaceBody, Source.intRange?, run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, FP.Structural.store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, ha, hz, hnonneg, hj0,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model, ReversedCumsum.TypedExecution.model,
    ofFloat, toFloat, Region.cast, Ready]

#axiomsClean body_success
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.TypedExecution

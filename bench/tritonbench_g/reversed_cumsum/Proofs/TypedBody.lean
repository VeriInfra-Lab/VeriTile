import bench.tritonbench_g.reversed_cumsum.Proofs.TypedReady

/-! Every actual loop statement succeeds in the mathematical consistency
backend. The input and output storage carriers are independently bf16/fp16;
tile shapes and loop index remain arbitrary. Dot and reduction values remain
symbolic, while both boundary checks and every storage cast execute. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite RealProgram
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem body_success (p : Layout) (s o : RegionName)
    (u : State (WithBot ℝ)) (j : Int) (hj : j < 1) (h : Ready p u) :
    ∃ v, run (engine model) (TypedProgram.parts Bool.false .bf16 .fp16 p s o).body
      (u.setReg "__rev_i_t" .int [] (fun _ => j)) = some v ∧ Ready p v := by
  rcases h with ⟨⟨a,ha⟩,⟨b,hb⟩,⟨m,hm⟩,⟨z,hz⟩⟩
  have hnonneg : 0 ≤ -j := by omega
  have hj0 : j ≤ 0 := by omega
  simp [TypedProgram.parts, TypedSource.source, TypedSource.original,
    ComputeKernel.surfaceBody, Source.intRange?, run, step, FP.Structural.evalExpr,
    evalComputeOp, evalOp_unfold, evalOffsets, FP.Structural.store, Option.bind, ComputeDType.eraseDType,
    numeric, bop, numericLe, ha, hb, hm, hz, hnonneg, hj0,
    engine, Algebra.withDefaultPrecision, resolvePrecision, model,
    ofFloat, toFloat, Region.cast, Ready]

#axiomsClean body_success
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution

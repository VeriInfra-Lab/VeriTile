import bench.tritonbench_g.reversed_cumsum.Proofs.TypedProgram

/-! Register availability for the full typed program, with arbitrary floating
tiles. It imposes no precomputed input slices, carries or output values. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite RealProgram
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def Ready (p : Layout) (u : State (WithBot ℝ)) : Prop :=
  (∃ v, u.regs .nat [] "i_s" = some v) ∧
  (∃ v, u.regs .nat [] "i_bh" = some v) ∧
  (∃ v, u.regs .real [p.BT,p.BT] "m_s" = some v) ∧
  (∃ v, u.regs .real [p.BS] "b_z" = some v)

theorem prefix_success (swapped : Bool) (dtype out_dtype : FloatDType)
    (p : Layout) (s o : RegionName) (u : State (WithBot ℝ)) :
    ∃ v, run (engine model) (TypedProgram.parts swapped dtype out_dtype p s o).before u =
      some v ∧ Ready p v := by
  cases swapped <;>
    simp [TypedProgram.parts, TypedSource.source, TypedSource.original, TypedSource.compareRight,
      ComputeKernel.surfaceBody, Source.intRange?, run, step, FP.Structural.evalExpr,
      evalComputeOp, evalOp_unfold, Option.bind, ComputeDType.eraseDType,
      numeric, bop, natLe, numericLe, Region.cast, Ready]

theorem mirror_ready (p : Layout) (u : State (WithBot ℝ)) (j : Int)
    (h : Ready p u) : Ready p (u.setReg "__rev_i_t" .int [] (fun _ => j)) := by
  rcases h with ⟨⟨a,ha⟩,⟨b,hb⟩,⟨m,hm⟩,⟨z,hz⟩⟩
  refine ⟨⟨a, ?_⟩,⟨b, ?_⟩,⟨m, ?_⟩,⟨z, ?_⟩⟩ <;> simp [ha,hb,hm,hz]

#axiomsClean prefix_success
#axiomsClean mirror_ready
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.TypedExecution

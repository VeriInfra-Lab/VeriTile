import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackwardReady

/-! The actual typed middle section retains the debug barrier, six pointer
rebases, and d_h zero initialization, including T=0. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1600000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem middle_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (u : State α) (hu : FirstReady p u) :
    ∃ v, run A (Program.backwardSecond Bool.false .fp32 .fp32 p r hasSeed).before u =
      some v ∧ SecondReady p v := by
  rcases hu with ⟨⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨b,hb⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨d,hd⟩,⟨q,hq⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨h,hh⟩⟩
  simp [Program.backwardSecond,Program.backwardFirst,Program.backward,TypedSource.backward,
    TypedSource.backwardOriginal,ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,
    FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,Region.cast,hik,hiv,hih,hb,hk,hv,hd,hq,hmk,hmv,hh,SecondReady]

#axiomsClean middle_success
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward

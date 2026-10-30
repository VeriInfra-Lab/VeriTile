import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedModel

/-! The original typed final branch runs even after a zero-length loop.
Both the actual final pointer expression and masked store are retained. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1200000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem final_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) (hu : Ready p u) :
    ∃ v, run A (Program.forwardParts Bool.false .fp32 .fp32 p r hasSeed storeFinal).after u =
      some v := by
  rcases hu with ⟨⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨b,hb⟩,⟨q,hq⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨o,ho⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨mkv,hmkv⟩,⟨h,hh⟩⟩
  cases storeFinal <;>
    simp [Program.forwardParts,Program.forward,TypedSource.forward,TypedSource.forwardOriginal,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,ComputeDType.eraseDType,numeric,bop,
      Region.cast,hik,hiv,hih,hmkv,hh]

#axiomsClean final_success
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution

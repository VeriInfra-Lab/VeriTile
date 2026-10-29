import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedPrefix

/-! Actual typed loop execution, including all masked loads, reduction,
output store and pointer increments. Floating tiles remain symbolic outputs
of the selected algebra, so no precomputed carry or input tiles are assumed. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem body_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) (t : Nat) (hu : Ready p u) :
    ∃ v, run A (Program.forwardParts Bool.false .fp32 .fp32 p r hasSeed storeFinal).body
      (u.setReg "_t" .nat [] (fun _ => t)) = some v ∧ Ready p v := by
  rcases hu with ⟨⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨b,hb⟩,⟨q,hq⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨o,ho⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨mkv,hmkv⟩,⟨h,hh⟩⟩
  simp [Program.forwardParts,Program.forward,TypedSource.forward,TypedSource.forwardOriginal,
    ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,ComputeDType.eraseDType,numeric,bop,
    Region.cast,hik,hiv,hih,hb,hq,hk,hv,ho,hmk,hmv,hmkv,hh,Ready]

#axiomsClean body_success
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution

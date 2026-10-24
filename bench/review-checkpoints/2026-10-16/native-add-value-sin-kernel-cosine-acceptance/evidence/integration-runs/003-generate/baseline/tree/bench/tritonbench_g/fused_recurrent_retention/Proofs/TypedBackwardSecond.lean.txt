import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackwardReady

/-! Actual typed second-loop success, preserving both stores and all six
pointer subtractions. Floating values remain the selected algebra's results. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 2200000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem second_body_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (u : State α) (t : Nat) (hu : SecondReady p u) :
    ∃ v, run A (Program.backwardSecond Bool.false .fp32 .fp32 p r hasSeed).body
      (u.setReg "_t" .nat [] (fun _ => t)) = some v ∧ SecondReady p v := by
  rcases hu with ⟨⟨b,hb⟩,⟨q,hq⟩,⟨k,hk⟩,⟨v,hv⟩,⟨d,hd⟩,⟨dk,hdk⟩,
    ⟨dv,hdv⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨h,hh⟩⟩
  simp [Program.backwardSecond,Program.backwardFirst,Program.backward,TypedSource.backward,
    TypedSource.backwardOriginal,ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,
    FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,hb,hq,hk,hv,hd,hdk,hdv,hmk,hmv,hh,SecondReady]

#axiomsClean second_body_success
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward

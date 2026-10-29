import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackwardPrefix

/-! Successful execution of every actual first-loop operation, including the
masked d_q store and all four pointer advances, for arbitrary typed algebra. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1800000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem first_body_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (u : State α) (t : Nat) (hu : FirstReady p u) :
    ∃ v, run A (Program.backwardFirst Bool.false .fp32 .fp32 p r hasSeed).body
      (u.setReg "i" .nat [] (fun _ => t)) = some v ∧ FirstReady p v := by
  rcases hu with ⟨⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨b,hb⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨d,hd⟩,⟨q,hq⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨h,hh⟩⟩
  simp [Program.backwardFirst,Program.backward,TypedSource.backward,TypedSource.backwardOriginal,
    ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,ComputeDType.eraseDType,numeric,bop,
    Region.cast,hik,hiv,hih,hb,hk,hv,hd,hq,hmk,hmv,hh,FirstReady]

#axiomsClean first_body_success
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward

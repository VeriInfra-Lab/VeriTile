import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.QuotedReady

/-! Successful execution of actual repaired-source statements. Only the
external scale register is assumed; all carries and loaded values are produced
by the shared interpreter. Native operation refinement is separate. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
theorem forward_prefix {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) (scale : Values α .real [])
    (hscale : u.regs .real [] "scale" = some scale) :
    ∃ v, run A (Program.forwardParts p r hasSeed storeFinal).before u =
      some v ∧ ForwardReady p v := by
  cases hasSeed <;>
    simp [Program.forwardParts,Program.forward,Surface.forward,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralMod,Region.cast,hscale,ForwardReady]

#axiomsClean forward_prefix

theorem forward_body {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) (t : Nat) (hu : ForwardReady p u) :
    ∃ v, run A (Program.forwardParts p r hasSeed storeFinal).body
      (u.setReg "_t" .nat [] (fun _ => t)) = some v ∧ ForwardReady p v := by
  rcases hu with ⟨⟨scale,hscale⟩,⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨b,hb⟩,⟨q,hq⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨o,ho⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨mkv,hmkv⟩,⟨h,hh⟩⟩
  simp [Program.forwardParts,Program.forward,Surface.forward,
    ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,ComputeDType.eraseDType,numeric,bop,
    Region.cast,hscale,hik,hiv,hih,hb,hq,hk,hv,ho,hmk,hmv,hmkv,hh,ForwardReady]

#axiomsClean forward_body

theorem forward_final {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) (hu : ForwardReady p u) :
    ∃ v, run A (Program.forwardParts p r hasSeed storeFinal).after u =
      some v := by
  rcases hu with ⟨⟨scale,hscale⟩,⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨b,hb⟩,⟨q,hq⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨o,ho⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨mkv,hmkv⟩,⟨h,hh⟩⟩
  cases storeFinal <;>
    simp [Program.forwardParts,Program.forward,Surface.forward,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,ComputeDType.eraseDType,numeric,bop,
      Region.cast,hscale,hik,hiv,hih,hmkv,hh]

#axiomsClean forward_final

theorem backward_prefix {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed hasFinal : Bool) (u : State α) (scale : Values α .real [])
    (hscale : u.regs .real [] "scale" = some scale) :
    ∃ v, run A (Program.backwardFirst p r hasSeed hasFinal).before u =
      some v ∧ FirstReady p v := by
  cases hasSeed <;>
    simp [Program.backwardFirst,Program.backward,Surface.backward,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralMod,Region.cast,hscale,FirstReady]

#axiomsClean backward_prefix

theorem backward_first {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed hasFinal : Bool) (u : State α) (t : Nat) (hu : FirstReady p u) :
    ∃ v, run A (Program.backwardFirst p r hasSeed hasFinal).body
      (u.setReg "i" .nat [] (fun _ => t)) = some v ∧ FirstReady p v := by
  rcases hu with ⟨⟨scale,hscale⟩,⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨head,hhead⟩,⟨b,hb⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨d,hd⟩,⟨q,hq⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨h,hh⟩⟩
  simp [Program.backwardFirst,Program.backward,Surface.backward,
    ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
    evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,ComputeDType.eraseDType,numeric,bop,
    Region.cast,hscale,hik,hiv,hih,hhead,hb,hk,hv,hd,hq,hmk,hmv,hh,FirstReady]

#axiomsClean backward_first

theorem backward_middle {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed hasFinal : Bool) (u : State α) (hu : FirstReady p u) :
    ∃ v, run A (Program.backwardSecond p r hasSeed hasFinal).before u =
      some v ∧ SecondReady p v := by
  rcases hu with ⟨⟨scale,hscale⟩,⟨ik,hik⟩,⟨iv,hiv⟩,⟨ih,hih⟩,⟨head,hhead⟩,⟨b,hb⟩,⟨k,hk⟩,⟨v,hv⟩,
    ⟨d,hd⟩,⟨q,hq⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨h,hh⟩⟩
  cases hasFinal <;> simp [Program.backwardSecond,Program.backwardFirst,Program.backward,Surface.backward,ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,
    FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
    numeric,bop,Region.cast,hscale,hik,hiv,hih,hhead,hb,hk,hv,hd,hq,hmk,hmv,hh,SecondReady]

#axiomsClean backward_middle

theorem backward_second {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed hasFinal : Bool) (u : State α) (t : Nat) (hu : SecondReady p u) :
    ∃ v, run A (Program.backwardSecond p r hasSeed hasFinal).body
      (u.setReg "_t" .nat [] (fun _ => t)) = some v ∧ SecondReady p v := by
  rcases hu with ⟨⟨scale,hscale⟩,⟨b,hb⟩,⟨q,hq⟩,⟨k,hk⟩,⟨v,hv⟩,⟨d,hd⟩,⟨dk,hdk⟩,
    ⟨dv,hdv⟩,⟨mk,hmk⟩,⟨mv,hmv⟩,⟨h,hh⟩⟩
  simp [Program.backwardSecond,Program.backwardFirst,Program.backward,Surface.backward,ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,
    FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.store,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,hscale,hb,hq,hk,hv,hd,hdk,hdv,hmk,hmv,hh,SecondReady]

#axiomsClean backward_second

end
end VeriTile.Bench.Repairs.FusedRecurrentRetention.Quoted

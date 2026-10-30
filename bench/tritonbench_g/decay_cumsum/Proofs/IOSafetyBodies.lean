import bench.tritonbench_g.decay_cumsum.Proofs.IOSafetyPrefix

namespace VeriTile.Bench.Repairs.DecayCumsum.IOSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPPrefix FPOps Geometry
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forward_safe (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (initial : State α) (n : Nat) (u : State α) (inv : FPForward.Holds M h p prog initial n u)
    (bounds : RegionName → Nat)
    (hg : ∀ j : Fin h.BK, off h p prog n j<bounds "g")
    (ho : ∀ j : Fin h.BK, off h p prog n j<bounds "go") :
    Safety.traceSafeList (engine M) bounds (forwardBody h) u := by
  simp (config := { maxSteps := 600000 }) [forwardBody,assign32,gateOp,loadOp,pointerOp,maskOp,
    fwdOp,refOp,storeCode,advanceCode,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
    inv.sourcePtr,inv.targetPtr,inv.mask,inv.acc,ptr,
    State.setReg,engine,numeric,bop,ComputeDType.eraseDType,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  exact ⟨hg,ho⟩

 theorem prepare_safe (M : Algebra α) (h : Call) (p : Phase) (prog : Program h p)
    (scale : ℝ) (initial : State α) (n : Nat) (u : State α)
    (inv : FPPrepare.Holds M h p prog scale initial n u) (bounds : RegionName → Nat)
    (hq : ∀ j : Fin h.BK, off h p prog n j<bounds "q")
    (hk : ∀ j : Fin h.BK, off h p prog n j<bounds "k")
    (hg : ∀ j : Fin h.BK, off h p prog n j<bounds "g")
    (hqo : ∀ j : Fin h.BK, off h p prog n j<bounds "qg")
    (hko : ∀ j : Fin h.BK, off h p prog n j<bounds "kg") :
    Safety.traceSafeList (engine M) bounds (prepareBody h scale) u := by
  simp (config := { maxSteps := 800000 }) [prepareBody,dataLoadCode,assign32,gateOp,loadOp,pointerOp,maskOp,
    prepareQOp,differenceGatedOp,refOp,storeCode,advanceCode,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
    inv.qPtr,inv.kPtr,inv.gPtr,inv.qoPtr,inv.koPtr,inv.mask,inv.last,ptr,
    State.setReg,engine,numeric,bop,ComputeDType.eraseDType,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  exact ⟨hq,hk,hg,hko,hqo⟩

#axiomsClean forward_safe
#axiomsClean prepare_safe
end
end VeriTile.Bench.Repairs.DecayCumsum.IOSafety

import bench.repairs.tritonbench_g.decay_cumsum.Proofs.IOSafetyPrefix

namespace VeriTile.Bench.Repairs.DecayCumsum.IOSafety
open VeriTile.Triton FP.Structural FP.ProfiledRewrite FPPrefix FPOps Geometry
open FPBackwardBase
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem backward_control_safe (M : Algebra α) (h : Call) (p : Phase) (n : Nat)
    (addr : Fin h.BK → Nat) (u : State α)
    (hi : u.regs .int [] "__rev_j"=some (fun _ => Int.ofNat n-Int.ofNat (p.span-1)))
    (hp : u.regs .ptr [h.BK] "p_g"=some (ptr h.BK "g" addr))
    (hm : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true))
    (bounds : RegionName → Nat) (hb : ∀ j, addr j<bounds "g") :
    Safety.traceSafeList (engine M) bounds (FPBackwardControl.firstCode h p) u := by
  simp (config := { maxSteps := 600000 }) [FPBackwardControl.firstCode,reverseIndex,assign32,
    gateOp,loadOp,pointerOp,maskOp,lastGate,refOp,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,run,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,hi,hp,hm,ptr,
    State.setReg,engine,numeric,numericEq,bop,ComputeDType.eraseDType,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,hb]
  split <;> simp

 theorem backward_arithmetic_safe (M : Algebra α) (h : Call) (addr : Fin h.BK → Nat)
    (g last acc : Values α .real [h.BK]) (u : State α)
    (pointers : ∀ name r, (name,r)∈pointerNames →
      u.regs .ptr [h.BK] name=some (ptr h.BK r addr))
    (hm : u.regs .bool [h.BK] "mask"=some (fun _ => Bool.true))
    (hg : u.regs .real [h.BK] "_g"=some g)
    (hl : u.regs .real [h.BK] "last_g"=some last)
    (ha : u.regs .real [h.BK] "cum_grad_dg"=some acc)
    (bounds : RegionName → Nat)
    (hb : ∀ r∈inputRegions++outputRegions, ∀ j, addr j<bounds r) :
    Safety.traceSafeList (engine M) bounds (backwardArithmetic h) u := by
  have pq := pointers "p_q" "q" (by simp [pointerNames])
  have pk := pointers "p_k" "k" (by simp [pointerNames])
  have pg := pointers "p_g" "g" (by simp [pointerNames])
  have pqi := pointers "p_dq_inner" "dqi" (by simp [pointerNames])
  have pki := pointers "p_dk_inner" "dki" (by simp [pointerNames])
  have pqo := pointers "p_dq_inter" "dqo" (by simp [pointerNames])
  have pko := pointers "p_dk_inter" "dko" (by simp [pointerNames])
  have pdg := pointers "p_dg" "dg" (by simp [pointerNames])
  have bq := hb "q" (by simp [inputRegions,outputRegions])
  have bk := hb "k" (by simp [inputRegions,outputRegions])
  have bqi := hb "dqi" (by simp [inputRegions,outputRegions])
  have bki := hb "dki" (by simp [inputRegions,outputRegions])
  have bqo := hb "dqo" (by simp [inputRegions,outputRegions])
  have bko := hb "dko" (by simp [inputRegions,outputRegions])
  have bdg := hb "dg" (by simp [inputRegions,outputRegions])
  simp (config := { maxSteps := 1000000 }) [backwardArithmetic,dataLoadCode,assign32,
    gateOp,loadOp,pointerOp,maskOp,gatedOp,differenceGatedOp,addOp,gradientOp,
    refOp,storeCode,retreatCode,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,store,
    pq,pk,pg,pqi,pki,pqo,pko,pdg,hm,hg,hl,ha,ptr,
    State.setReg,engine,numeric,bop,ComputeDType.eraseDType,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast]
  tauto

#axiomsClean backward_control_safe
#axiomsClean backward_arithmetic_safe
end
end VeriTile.Bench.Repairs.DecayCumsum.IOSafety

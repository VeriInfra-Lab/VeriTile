import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelBackward
import VeriTile.Triton.Float.StructuralSafety
import bench.tritonbench_g.isfinite_kernel.Proofs.Mathematical_MemoryProjection
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters KernelBackwardRegisters
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Precisely the forward-readable and gradient read/write windows of the
actual backward quotation. Public IO bounds discharge these conditions. -/
structure Bounds (s : Shape) (f : Flags) (bh iv : Nat) (bounds : RegionName → Nat) : Prop where
  query : ∀ t,t<s.T → ∀ i : Fin s.BK,i.val<s.K → keyAddress s bh t i.val<bounds "q"
  key : ∀ t,t<s.T → ∀ i : Fin s.BK,i.val<s.K → keyAddress s bh t i.val<bounds "k"
  value : ∀ t,t<s.T → ∀ j : Fin s.BBV,iv*s.BBV+j.val<s.V →
    valAddress s bh iv s.BBV t j.val<bounds "v"
  incoming : ∀ t,t<s.T → ∀ j : Fin s.BBV,iv*s.BBV+j.val<s.V →
    valAddress s bh iv s.BBV t j.val<bounds "do"
  betaVector : f.vectorBeta=Bool.true → ∀ t,t<s.T → ∀ j : Fin s.BBV,iv*s.BBV+j.val<s.V →
    valAddress s bh iv s.BBV t j.val<bounds "beta"
  betaScalar : f.vectorBeta=Bool.false → ∀ t,t<s.T → betaAddress s bh t<bounds "beta"
  dquery : ∀ t,t<s.T → ∀ i : Fin s.BK,i.val<s.K → keyAddress s (plane s bh iv) t i.val<bounds "dq"
  dkey : ∀ t,t<s.T → ∀ i : Fin s.BK,i.val<s.K → keyAddress s (plane s bh iv) t i.val<bounds "dk"
  dvalue : ∀ t,t<s.T → ∀ j : Fin s.BBV,iv*s.BBV+j.val<s.V →
    valAddress s bh iv s.BBV t j.val<bounds "dv"
  dbetaVector : f.vectorBeta=Bool.true → ∀ t,t<s.T → ∀ j : Fin s.BBV,iv*s.BBV+j.val<s.V →
    valAddress s (plane s bh iv) iv s.BBV t j.val<bounds "dbeta"
  dbetaScalar : f.vectorBeta=Bool.false → ∀ t,t<s.T → plane s bh iv*s.T+t<bounds "dbeta"
  initial : f.initial=Bool.true → ∀ i : Fin s.BK,∀ j : Fin s.BBV,
    i.val<s.K → iv*s.BBV+j.val<s.V → stateAddress s bh iv s.BBV i.val j.val<bounds "h0"
  terminal : f.terminalGrad=Bool.true → ∀ i : Fin s.BK,∀ j : Fin s.BBV,
    i.val<s.K → iv*s.BBV+j.val<s.V → stateAddress s bh iv s.BBV i.val j.val<bounds "dht"
  seed : f.initialGrad=Bool.true → ∀ i : Fin s.BK,∀ j : Fin s.BBV,
    i.val<s.K → iv*s.BBV+j.val<s.V → stateAddress s bh iv s.BBV i.val j.val<bounds "dh0"

theorem prefix_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ)
    (bh iv : Nat) (u : State α) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.traceSafeList (engine M) bounds (first .backward s f scale).before u := by
  cases ht : f.terminalGrad <;> cases hb : f.vectorBeta
  all_goals
    simp only [first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,ht,hb,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=350000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
      FP.Structural.store,integralFloorDiv,integralMod,hv,hk,hh,
      keyPtr,valPtr,betaPtr,dbetaPtr,keyMask,valMask,fold_writes_regs,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  all_goals
    intro i j key value
    simpa [stateAddress,Nat.add_assoc] using access.terminal ht i j key value

theorem reverse_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ)
    (bh iv t : Nat) (inside : t<s.T) (h : Tile s α) (u : State α)
    (regs : KernelBackwardRegisters.Reverse s f bh iv t h u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.traceSafeList (engine M) bounds (first .backward s f scale).body u := by
  obtain ⟨hq,hk,hv,hdo,hdk,hdv,hb,hdb,hmk,hmv,hh⟩ := regs
  cases hbv : f.vectorBeta
  all_goals
    simp only [hbv,Bool.false_eq_true,ite_false,ite_true] at hb hdb
    simp only [second,first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hbv,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=350000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
      FP.Structural.store,integralFloorDiv,integralMod,hq,hk,hv,hdo,hdk,hdv,hb,hdb,hmk,hmv,hh,
      keyPtr,valPtr,betaPtr,dbetaPtr,keyMask,valMask,fold_writes_regs,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
    repeat' (first | apply And.intro | intro)
    all_goals first
    | exact access.query t inside _ (by assumption)
    | exact access.key t inside _ (by assumption)
    | exact access.value t inside _ (by assumption)
    | exact access.incoming t inside _ (by assumption)
    | exact access.dquery t inside _ (by assumption)
    | exact access.dkey t inside _ (by assumption)
    | exact access.dvalue t inside _ (by assumption)
    | exact access.betaScalar hbv t inside
    | exact access.betaVector hbv t inside _ (by assumption)
    | exact access.dbetaScalar hbv t inside
    | exact access.dbetaVector hbv t inside _ (by assumption)

theorem recompute_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ)
    (bh iv t : Nat) (inside : t<s.T) (h : Tile s α) (u : State α)
    (regs : KernelBackwardRegisters.Recompute s f bh iv t h u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.traceSafeList (engine M) bounds (second s f scale).body u := by
  obtain ⟨hk,hv,hdo,hdq,hdk,hdv,hb,hmk,hmv,hh⟩ := regs
  cases hbv : f.vectorBeta
  all_goals
    simp only [hbv,Bool.false_eq_true,ite_false,ite_true] at hb
    simp only [second,first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hbv,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=350000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
      FP.Structural.store,integralFloorDiv,integralMod,hk,hv,hdo,hdq,hdk,hdv,hb,hmk,hmv,hh,
      keyPtr,valPtr,betaPtr,dbetaPtr,keyMask,valMask,fold_writes_regs,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
    repeat' (first | apply And.intro | intro)
    all_goals first
    | exact access.query t inside _ (by assumption)
    | exact access.key t inside _ (by assumption)
    | exact access.value t inside _ (by assumption)
    | exact access.incoming t inside _ (by assumption)
    | exact access.dquery t inside _ (by assumption)
    | exact access.dkey t inside _ (by assumption)
    | exact access.dvalue t inside _ (by assumption)
    | exact access.betaScalar hbv t inside
    | exact access.betaVector hbv t inside _ (by assumption)
    | exact access.dbetaScalar hbv t inside
    | exact access.dbetaVector hbv t inside _ (by assumption)

theorem middle_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (h : Tile s α) (u : State α) (carry : u.regs .real [s.BK,s.BBV] "d_h"=some h)
    (static : KernelBackwardInitial.Static s bh iv u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.traceSafeList (engine M) bounds (second s f scale).before u := by
  obtain ⟨head,key,value,mk,mv⟩ := static
  cases hb : f.vectorBeta <;> cases hi : f.initial <;> cases hg : f.initialGrad
  all_goals
    simp only [second,first,source,Source.backward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hb,hi,hg,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=350000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
      FP.Structural.store,integralFloorDiv,integralMod,head,key,value,mk,mv,carry,
      keyPtr,valPtr,betaPtr,dbetaPtr,keyMask,valMask,fold_writes_regs,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
    repeat' (first | apply And.intro | intro)
    all_goals first
    | simpa [stateAddress,Nat.add_assoc] using access.initial hi _ _ (by assumption) (by assumption)
    | simpa [stateAddress,Nat.add_assoc] using access.seed hg _ _ (by assumption) (by assumption)

#axiomsClean prefix_safe
#axiomsClean reverse_safe
#axiomsClean recompute_safe
#axiomsClean middle_safe
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyBackward

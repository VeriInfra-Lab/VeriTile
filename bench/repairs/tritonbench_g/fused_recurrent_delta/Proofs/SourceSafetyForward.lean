import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelForward
import VeriTile.Triton.Float.StructuralSafety
import bench.tritonbench_g.isfinite_kernel.Proofs.MemoryProjection
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyForward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite KernelProgram KernelRegisters
open VeriTile.Bench.TritonBenchG.IsfiniteKernel.TypedBits (fold_writes_regs)
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The exact forward program's masked windows, including its in-place
residual output and optional state inputs/outputs. These are derived from the
public IO channels, not admitted as a public safety premise. -/
structure Bounds (s : Shape) (f : Flags) (bh iv : Nat) (bounds : RegionName → Nat) : Prop where
  query : ∀ t,t<s.T → ∀ i : Fin s.BK,i.val<s.K → keyAddress s bh t i.val<bounds "q"
  key : ∀ t,t<s.T → ∀ i : Fin s.BK,i.val<s.K → keyAddress s bh t i.val<bounds "k"
  value : ∀ t,t<s.T → ∀ j : Fin s.FBV,iv*s.FBV+j.val<s.V →
    valAddress s bh iv s.FBV t j.val<bounds "v"
  betaVector : f.vectorBeta=Bool.true → ∀ t,t<s.T → ∀ j : Fin s.FBV,iv*s.FBV+j.val<s.V →
    valAddress s bh iv s.FBV t j.val<bounds "beta"
  betaScalar : f.vectorBeta=Bool.false → ∀ t,t<s.T → betaAddress s bh t<bounds "beta"
  output : ∀ t,t<s.T → ∀ j : Fin s.FBV,iv*s.FBV+j.val<s.V →
    valAddress s bh iv s.FBV t j.val<bounds "o"
  initial : f.initial=Bool.true → ∀ i : Fin s.BK,∀ j : Fin s.FBV,
    i.val<s.K → iv*s.FBV+j.val<s.V → stateAddress s bh iv s.FBV i.val j.val<bounds "h0"
  terminal : f.final=Bool.true → ∀ i : Fin s.BK,∀ j : Fin s.FBV,
    i.val<s.K → iv*s.FBV+j.val<s.V → stateAddress s bh iv s.FBV i.val j.val<bounds "ht"

theorem prefix_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ)
    (bh iv : Nat) (u : State α) (hv : u.pids 0=iv) (hk : u.pids 1=0) (hh : u.pids 2=bh)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.traceSafeList (engine M) bounds (first .forward s f scale).before u := by
  cases hi : f.initial <;> cases hb : f.vectorBeta
  all_goals
    simp only [first,source,Source.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hi,hb,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=250000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      FP.Structural.store,integralFloorDiv,integralMod,hv,hk,hh,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex]
  all_goals
    intro j i key value
    simpa [stateAddress,Nat.add_assoc] using access.initial hi i j key value

theorem body_safe (M : Algebra α) (s : Shape) (f : Flags) (scale : ℝ)
    (bh iv t : Nat) (inside : t<s.T) (h : ForwardTile s α) (u : State α)
    (regs : KernelRegisters.Forward s f bh iv t h u)
    (bounds : RegionName → Nat) (access : Bounds s f bh iv bounds) :
    Safety.traceSafeList (engine M) bounds (first .forward s f scale).body u := by
  obtain ⟨hq,hk,hv,ho,hb,hmk,hmv,hh⟩ := regs
  cases hbv : f.vectorBeta
  all_goals
    simp only [hbv,Bool.false_eq_true,ite_false,ite_true] at hb
    simp only [first,source,Source.forward,ComputeKernel.surfaceBody,
      VeriTile.Triton.Source.dynamicRange?,Option.map,Option.getD,String.reduceEq,
      beq_self_eq_true,hbv,Bool.false_eq_true,ite_false,ite_true,↓reduceIte]
    simp (config := {maxSteps:=300000}) [Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      FP.Structural.run,FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,
      engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      FP.Structural.store,integralFloorDiv,integralMod,hq,hk,hv,ho,hb,hmk,hmv,hh,
      State.setReg_same,State.setReg_ne_name,State.setReg_mem,State.setReg_pids,State.setReg_undef,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,
      keyPtr,valPtr,betaPtr,keyMask,valMask,fold_writes_regs]
    repeat' (first | apply And.intro | intro)
    all_goals first
    | exact access.query t inside _ (by assumption)
    | exact access.key t inside _ (by assumption)
    | exact access.value t inside _ (by assumption)
    | exact access.output t inside _ (by assumption)
    | exact access.betaScalar hbv t inside
    | exact access.betaVector hbv t inside _ (by assumption)

#axiomsClean prefix_safe
#axiomsClean body_safe
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.SourceSafetyForward

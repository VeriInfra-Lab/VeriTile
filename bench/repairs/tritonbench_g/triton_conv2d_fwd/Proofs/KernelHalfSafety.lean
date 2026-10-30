import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.KernelHalfComplete
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.Conv2d.KernelHalf.Access
open VeriTile Triton FP.Structural KernelHalf
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Bounds for exactly the source's active loads and stores. Loop coordinates
are constrained by their actual source ranges; inactive pointers need no storage. -/
structure Bounds (p : Parameters) (s : State α) (bounds : RegionName → Nat) : Prop where
  input : ∀ h, h < p.kernel_height → ∀ w, w < p.kernel_width → ∀ c, c < igd p →
    ∀ i, inputMask p s h w c i → inputOffset p s h w c i  <  bounds p.Input
  weight : ∀ h, h < p.kernel_height → ∀ w, w < p.kernel_width → ∀ c, c < igd p →
    ∀ i, weightMask p s c i → weightOffset p s h w c i  <  bounds p.Weight
  output : ∀ i, active p s i → outputOffset p s i  <  bounds p.Output

theorem prelude_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList (backend M D) bounds (prelude p) s := by
  simp [prelude,source,halfBody,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeOp_unfold,Safety.safeCompute_unfold,
    Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,FP.Structural.store,
    numeric,bop,natLt,numericEq,numericLe,numericLt,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    Option.bind,integralFloorDiv,integralMod,backend,Broadcast.leftIndex,Broadcast.rightIndex]

theorem body_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (initial current : State α) (h w c : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (hh : current.regs .nat [] "h" = some (fun _ => h))
    (hw : current.regs .nat [] "w" = some (fun _ => w))
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds)
    (hhb : h < p.kernel_height) (hwb : w < p.kernel_width) (hcb : c < igd p) :
    Safety.traceSafeList (backend M D) bounds (cBody p)
      (current.setReg "c" .nat [] (fun _ => c)) := by
  simp only [HasAcc] at ha
  simp [cBody,wBody,hBody,source,halfBody,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeOp_unfold,Safety.safeCompute_unfold,
    Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,FP.Structural.store,
    numeric,bop,natLt,numericEq,numericLe,numericLt,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    Option.bind,integralFloorDiv,integralMod,backend,Broadcast.leftIndex,Broadcast.rightIndex,
    ctx.memory,ctx.undefined,ctx.group,ctx.ig,ctx.og,ctx.ba,ctx.fe,ctx.he,
    ctx.wi,ctx.ip,ctx.wp,ha,hh,hw]
  constructor
  · intro i k hn hc hy0 hy hx0 hx
    have hh : 0 ≤ (p.input_height_stride : Int) * ih p initial h i :=
      mul_nonneg (Int.natCast_nonneg _) hy0
    have hw : 0 ≤ (p.input_width_stride : Int) * iw p initial w i :=
      mul_nonneg (Int.natCast_nonneg _) hx0
    have hhn : (-((p.input_height_stride : Int) * ih p initial h i)).toNat = 0 :=
      Int.toNat_eq_zero.mpr (neg_nonpos.mpr hh)
    have hwn : (-((p.input_width_stride : Int) * iw p initial w i)).toNat = 0 :=
      Int.toNat_eq_zero.mpr (neg_nonpos.mpr hw)
    dsimp only [ih] at hhn
    dsimp only [iw] at hwn
    simpa only [inputOffset,ih,iw,hhn,hwn,Nat.sub_zero] using
      hb.input h hhb w hwb c hcb (i,k,()) ⟨⟨⟨⟨⟨hn,hc⟩,hy0⟩,hy⟩,hx0⟩,hx⟩
  · intro k j hc hf
    exact hb.weight h hhb w hwb c hcb (k,j,()) ⟨hc,hf⟩

theorem epilogue_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (initial current : State α) (ctx : Context p initial current)
    (ha : HasAcc p (accumulation M D p initial) current)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    Safety.traceSafeList (backend M D) bounds (epilogue p) current := by
  simp only [HasAcc] at ha
  simp [epilogue,source,halfBody,
    ComputeKernel.surfaceBody,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeOp_unfold,Safety.safeCompute_unfold,
    Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,FP.Structural.store,
    numeric,bop,natLt,numericEq,numericLe,numericLt,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,
    Option.bind,integralFloorDiv,integralMod,backend,Broadcast.leftIndex,Broadcast.rightIndex,
    ctx.memory,ctx.group,ctx.og,ctx.ba,ctx.fe,ctx.he,ctx.wi,ha]
  intro i j hn hf hh hw
  exact hb.output (i,j,()) ⟨⟨⟨hn,hf⟩,hh⟩,hw⟩


theorem channel_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (initial current : State α) (h w : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (hh : current.regs .nat [] "h" = some (fun _ => h))
    (hw : current.regs .nat [] "w" = some (fun _ => w))
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds)
    (hhb : h < p.kernel_height) (hwb : w < p.kernel_width) :
    Safety.traceSafeList (backend M D) bounds (wBody p) current := by
  rw [wBody_shape,Safety.traceSafeList_cons]
  refine ⟨?_,fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,ctx.ig,true_and]
  apply Safety.rangeSafe_of_invariant (backend M D) bounds "c" (igd p) (BIN p) (cBody p)
    (fun _ t => ∃ a, Context p initial t ∧ HasAcc p a t ∧
      t.regs .nat [] "h"=some (fun _ => h) ∧ t.regs .nat [] "w"=some (fun _ => w))
  · rintro c hc t ⟨a,ctx,ha,hh,hw⟩
    exact body_safe M D p initial t h w c a ctx ha hh hw bounds hb hhb hwb hc
  · rintro c hc t ⟨a,ctx,ha,hh,hw⟩ u hu
    have eq := Option.some.inj ((bodyExecution M D p initial t h w c a ctx ha hh hw).property.symm.trans hu)
    subst u
    exact ⟨update M D p initial h w c a,bodyContext M D p initial t h w c a ctx ha hh hw⟩
  · exact ⟨a,ctx,ha,hh,hw⟩

theorem width_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (initial current : State α) (h : Nat) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (hh : current.regs .nat [] "h" = some (fun _ => h))
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds)
    (hhb : h < p.kernel_height) :
    Safety.traceSafeList (backend M D) bounds (hBody p) current := by
  rw [hBody_shape,Safety.traceSafeList_cons]
  refine ⟨?_,fun _ _ => Safety.traceSafeList_nil _ _ _⟩
  simp only [Safety.stepSafe_unfold, Safety.loopSafe_eq_rangeSafe]
  apply Safety.rangeSafe_of_invariant (backend M D) bounds "w" p.kernel_width 1 (wBody p)
    (fun _ t => ∃ a, Context p initial t ∧ HasAcc p a t ∧ t.regs .nat [] "h"=some (fun _ => h))
  · rintro w hw t ⟨a,ctx,ha,hh⟩
    exact channel_safe M D p initial (t.setReg "w" .nat [] (fun _ => w)) h w a
      (ctx.set_loop p initial t "w" w (Or.inr (Or.inl rfl)))
      (by simpa using ha) (by simpa using hh) (by simp) bounds hb hhb hw
  · rintro w hw t ⟨a,ctx,ha,hh⟩ u hu
    obtain ⟨u',he,hc,ha',hh'⟩ := channelLoop M D p initial (t.setReg "w" .nat [] (fun _ => w)) h w a
      (ctx.set_loop p initial t "w" w (Or.inr (Or.inl rfl)))
      (by simpa using ha) (by simpa using hh) (by simp)
    have eq := Option.some.inj (he.symm.trans hu)
    subst u
    exact ⟨channels M D p initial h w a,hc,ha',hh'⟩
  · exact ⟨a,ctx,ha,hh⟩

theorem height_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (initial current : State α) (a : CIndex p → α)
    (ctx : Context p initial current) (ha : HasAcc p a current)
    (bounds : RegionName → Nat) (hb : Bounds p initial bounds) :
    Safety.stepSafe (backend M D) bounds (.forLoop "h" p.kernel_height (hBody p)) current := by
  simp only [Safety.stepSafe_unfold, Safety.loopSafe_eq_rangeSafe]
  apply Safety.rangeSafe_of_invariant (backend M D) bounds "h" p.kernel_height 1 (hBody p)
    (fun _ t => ∃ a, Context p initial t ∧ HasAcc p a t)
  · rintro h hh t ⟨a,ctx,ha⟩
    exact width_safe M D p initial (t.setReg "h" .nat [] (fun _ => h)) h a
      (ctx.set_loop p initial t "h" h (Or.inl rfl)) (by simpa using ha) (by simp) bounds hb hh
  · rintro h hh t ⟨a,ctx,ha⟩ u hu
    obtain ⟨u',he,hc,ha'⟩ := widthLoop M D p initial (t.setReg "h" .nat [] (fun _ => h)) h a
      (ctx.set_loop p initial t "h" h (Or.inl rfl)) (by simpa using ha) (by simp)
    have eq := Option.some.inj (he.symm.trans hu)
    subst u
    exact ⟨widths M D p initial h a,hc,ha'⟩
  · exact ⟨a,ctx,ha⟩

/-- The actual nested source loops and both their predecessor memories and
masked addresses satisfy the shared Structural safety judgment. -/
theorem source_safe (M : Algebra α) (D : Dot α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    TraceSafe (backend M D) bounds (source p) s := by
  change Safety.traceSafeList _ bounds (source p).surfaceBody s
  rw [decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe M D p s bounds,?_⟩
  intro t ran
  have eq := Option.some.inj ((preludeExecution M D p s).property.symm.trans ran)
  subst t
  obtain ⟨ctx,ha⟩ := preludeContext M D p s
  rw [List.singleton_append,Safety.traceSafeList_cons]
  refine ⟨height_safe M D p s _ _ ctx ha bounds hb,?_⟩
  intro u hu
  obtain ⟨v,hv,hc,ha'⟩ := heightLoop M D p s _ _ ctx ha
  have hx : step (backend M D) (.forLoop "h" p.kernel_height (hBody p))
      (preludeExecution M D p s).val=some v := by simpa [run] using hv
  have eq := Option.some.inj (hx.symm.trans hu)
  subst u
  exact epilogue_safe M D p s v hc ha' bounds hb

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.Conv2d.KernelHalf.Access

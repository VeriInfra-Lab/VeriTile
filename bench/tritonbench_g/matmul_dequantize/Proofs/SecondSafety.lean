import bench.tritonbench_g.matmul_dequantize.Proofs.SecondComplete
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.MatmulDequantize.SecondAccess
open VeriTile Triton FP.Structural
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open FPTypedSecond
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

/-- All four input loads retain the original K mask and pointer formulas. -/
structure Bounds (p : Parameters) (s : State A) (bounds : RegionName → Nat) : Prop where
  input : ∀ k,k<p.steps → ∀ i : AIndex p,innerActive p s k i.2.1 → aOffset p s k i<bounds p.input
  weight : ∀ k,k<p.steps → ∀ i : BIndex p,innerActive p s k i.1 → bOffset p s k i<bounds p.weight
  scale : ∀ k,k<p.steps → ∀ i : BIndex p,innerActive p s k i.1 → scaleOffset p s k i<bounds p.scale
  zero : ∀ k,k<p.steps → ∀ i : BIndex p,innerActive p s k i.1 → zeroOffset p s k i<bounds p.zero
  output : ∀ i,active p s i → outputOffset p s i<bounds p.output

private theorem min_form (a b : Nat) : min a b = if a<b then a else b := by
  split <;> omega

theorem prelude_safe (M : Algebra A) (D : Dot A) (p : Parameters) (s : State A)
    (bounds : RegionName → Nat) : Safety.traceSafeList (backend M D) bounds (prelude p) s := by
  simp [prelude,source,FPTyped.matmulTyped,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,FP.Structural.store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex]

theorem body_safe (M : Algebra A) (D : Dot A) (p : Parameters) (s t : State A)
    (k : Nat) (hk : k<p.steps) (h : Invariant M D p s k t)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (backend M D) bounds (body p) (t.setReg "k" .nat [] (fun _ => k)) := by
  simp [body,source,FPTyped.matmulTyped,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,FP.Structural.store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    h.offsk,h.offsn,h.aptrs,h.bptrs,h.acc,h.memory,backend]
  refine ⟨fun i j u hi => hb.input k hk (i,j,u) hi,
    fun i j u hi => hb.weight k hk (i,j,u) hi,fun i j hi => ?_,fun i j hi => ?_⟩
  · exact hb.scale k hk (i,j,()) hi
  · exact hb.zero k hk (i,j,()) hi

theorem epilogue_safe (M : Algebra A) (D : Dot A) (p : Parameters) (s t : State A)
    (h : Invariant M D p s (p.steps) t) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (backend M D) bounds (epilogue p) t := by
  simp [epilogue,source,FPTyped.matmulTyped,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
    evalOffsets,FP.Structural.store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
    ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    h.pidm,h.pidn,h.acc,h.memory,backend]
  intro i j hc hr
  exact hb.output (i,j,()) ⟨hc,hr⟩

theorem source_safe (M : Algebra A) (D : Dot A) (p : Parameters) (s : State A)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    TraceSafe (backend M D) bounds (source p) s := by
  change Safety.traceSafeList _ bounds (source p).surfaceBody s
  rw [source_decomposition,List.append_assoc,Safety.traceSafeList_append]
  refine ⟨prelude_safe M D p s bounds,?_⟩
  intro t ran
  have eq := Option.some.inj ((preludeExecution M D p s).property.symm.trans ran)
  subst t
  have initial := preludeInvariant M D p s
  rw [List.singleton_append,Safety.traceSafeList_cons]
  constructor
  · simp only [Safety.stepSafe_unfold]
    apply Safety.rangeSafe_of_invariant (backend M D) bounds "k" (p.steps) 1 (body p)
      (Invariant M D p s)
    · intro k hk t h
      exact body_safe M D p s t k hk h bounds hb
    · intro k hk t h u hu
      have eq := Option.some.inj ((bodyExecution M D p s t k h).property.symm.trans hu)
      subst u
      exact bodyInvariant M D p s t k h
    · exact initial
  · intro u hu
    obtain ⟨v,hv,hi⟩ := loopExecution M D p s (preludeExecution M D p s).val 0 (by omega) initial
    have runeq : step (backend M D)
        (.forRange "k" 0 p.steps 1 (body p))
        (preludeExecution M D p s).val=some v := by
      simpa [step] using hv
    have eq := Option.some.inj (runeq.symm.trans hu)
    subst u
    exact epilogue_safe M D p s v hi bounds hb

#axiomsClean source_safe
#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.MatmulDequantize.SecondAccess

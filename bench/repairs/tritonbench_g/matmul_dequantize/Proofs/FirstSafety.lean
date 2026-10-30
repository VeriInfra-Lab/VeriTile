import bench.tritonbench_g.matmul_dequantize.Proofs.FPFirstComplete
import VeriTile.Triton.Float.StructuralSafetyLoops

namespace VeriTile.Bench.Repairs.MatmulDequantize.FirstAccess
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulDequantize
open FPTypedFirst
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

/-- Actual source windows. Ungrouped scale/zero preloads occur even for zero
K iterations; A has only a row mask, and packed weight reads are unmasked. -/
structure Bounds (p : Parameters) (s : State A) (bounds : RegionName → Nat) : Prop where
  input : ∀ k,k<steps p → ∀ i : AIndex p,inputRow p s i.1<p.rows → aOffset p s k i<bounds p.input
  weight : ∀ k,k<steps p → ∀ i : BIndex p,bOffset p s k i<bounds p.weight
  scale : ∀ k,(k<steps p ∨ p.noGroups=Bool.true ∧ k=0) → ∀ i : TileIndex [p.blockColumns],
    scaleOffset p s k i<bounds p.scale
  zero : ∀ k,(k<steps p ∨ p.noGroups=Bool.true ∧ k=0) → ∀ i : TileIndex [p.blockColumns],
    zeroOffset p s k i<bounds p.zero
  output : ∀ i,active p s i → outputOffset p s i<bounds p.output

private theorem min_form (a b : Nat) : min a b = if a<b then a else b := by
  split <;> omega

theorem prelude_safe (M : Algebra A) (D : Dot A) (p : Parameters) (s : State A)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (backend M D) bounds (prelude p) s := by
  by_cases hn : p.noGroups=Bool.true
  · have scale := hb.scale 0 (Or.inr ⟨hn,rfl⟩)
    have zero := hb.zero 0 (Or.inr ⟨hn,rfl⟩)
    simp [scaleOffset,zeroOffset,inputColumn,columnBlock,rowsInGroup,firstRow,blocksM,blocksN,
      groupIndex,hn,min_form] at scale zero
    simp [prelude,source,FPTyped.matmul4Typed,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,FP.Structural.store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
      ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,hn,Broadcast.leftIndex,Broadcast.rightIndex]
    exact ⟨scale,zero⟩
  · simp [prelude,source,FPTyped.matmul4Typed,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,FP.Structural.store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
      ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,hn,Broadcast.leftIndex,Broadcast.rightIndex]

theorem body_safe (M : Algebra A) (D : Dot A) (p : Parameters) (s t : State A)
    (k : Nat) (hk : k<steps p) (h : Invariant M D p s k t)
    (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (backend M D) bounds (body p) (t.setReg "k" .nat [] (fun _ => k)) := by
  by_cases hn : p.noGroups=Bool.true
  all_goals
    simp [body,source,FPTyped.matmul4Typed,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeOp_unfold,Safety.safeCompute_unfold,Safety.safeAddress,Safety.safeMask,
      Safety.addressSafe,Safety.maskActive,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      evalOffsets,FP.Structural.store,numeric,bop,natLt,numericEq,ComputeDType.eraseDType,
      ofFloat,toFloat,FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,hn,Broadcast.leftIndex,Broadcast.rightIndex,
      h.aptrs,h.bptrs,h.acc,h.memory,h.amask,h.sptrs,h.zptrs,h.shift,h.zshift,
      h.bits,h.inf,h.scales,h.zeros,backend]
  · exact ⟨fun i j u hi => hb.input k hk (i,j,u) hi,fun i j u => hb.weight k hk (i,j,u)⟩
  · refine ⟨fun i j u hi => hb.input k hk (i,j,u) hi,
      fun i j u => hb.weight k hk (i,j,u),fun i => ?_,fun i => ?_⟩
    · simpa [scaleOffset,groupIndex,hn] using hb.scale k (Or.inl hk) (i,())
    · simpa [zeroOffset,groupIndex,hn] using hb.zero k (Or.inl hk) (i,())

theorem epilogue_safe (M : Algebra A) (D : Dot A) (p : Parameters) (s t : State A)
    (h : Invariant M D p s (steps p) t) (bounds : RegionName → Nat) (hb : Bounds p s bounds) :
    Safety.traceSafeList (backend M D) bounds (epilogue p) t := by
  simp [epilogue,source,FPTyped.matmul4Typed,ComputeKernel.surfaceBody,
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
  refine ⟨prelude_safe M D p s bounds hb,?_⟩
  intro t ran
  have eq := Option.some.inj ((preludeExecution M D p s).property.symm.trans ran)
  subst t
  have initial := preludeInvariant M D p s
  rw [List.singleton_append,Safety.traceSafeList_cons]
  constructor
  · simp only [Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold,initial.numk,
      Option.bind_some,true_and]
    apply Safety.rangeSafe_of_invariant (backend M D) bounds "k" (steps p) 1 (body p)
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
        (.forRangeDyn "k" (.constNat 0) (.ref .nat [] "num_pid_k") (.constNat 1) (body p))
        (preludeExecution M D p s).val=some v := by
      simpa [step,evalOp_unfold,initial.numk] using hv
    have eq := Option.some.inj (runeq.symm.trans hu)
    subst u
    exact epilogue_safe M D p s v hi bounds hb

#axiomsClean source_safe
#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.MatmulDequantize.FirstAccess

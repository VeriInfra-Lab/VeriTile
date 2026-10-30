import bench.repairs.tritonbench_g.matmul_triton2.revision001.Proofs.FPComplete
import VeriTile.Triton.Float.StructuralSafety

/-! Bounds for the actual selected grouped-grid source, including every
masked K iteration. Numerical dot semantics are opaque and irrelevant here. -/
namespace VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.KernelSafety
open VeriTile.Triton FP.Structural Revision001.Core
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prelude_safe (M : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend M P) bounds (prelude p) s := by
  simp [prelude,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod]

theorem body_safe (M : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (q : Nat)
    (ctx : Invariant P p initial q current)
    (ha : ∀ i : AIndex p, row p initial i.1<p.rows →
      i.2.1.val< inner p-q*p.blockInner → aOffset p initial q i<bounds p.input)
    (hb : ∀ i : BIndex p, i.1.val< inner p-q*p.blockInner →
      column p initial i.2.1<p.columns → bOffset p initial q i<bounds p.weight) :
    Safety.traceSafeList (backend M P) bounds (body p)
      (current.setReg "k" .nat [] (fun _ => q)) := by
  simp [body,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.offm,ctx.offn,ctx.offk,
    ctx.aptrs,ctx.bptrs,ctx.acc,backend]
  exact ⟨fun i j rest hr hk => ha (i,j,rest) hr hk,
    fun i j rest hk hc => hb (i,j,rest) hk hc⟩

theorem epilogue_safe (M : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α)
    (ctx : Invariant P p initial (steps p) current)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds p.output) :
    Safety.traceSafeList (backend M P) bounds (epilogue p) current := by
  simp [epilogue,source,typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.offm,ctx.offn,ctx.acc,backend]
  intro i j hr hn
  simpa [cOffset] using hc (i,j,PUnit.unit) ⟨hr,hn⟩

theorem range_safe (M : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State α) (q : Nat)
    (ctx : Invariant P p initial q current)
    (ha : ∀ q (i : AIndex p), row p initial i.1<p.rows →
      i.2.1.val< inner p-q*p.blockInner → aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : BIndex p), i.1.val< inner p-q*p.blockInner →
      column p initial i.2.1<p.columns → bOffset p initial q i<bounds p.weight) :
    Safety.rangeSafe (backend M P) bounds "k" q (steps p) 1 (body p) current := by
  by_cases inside : q<steps p
  · rw [Safety.rangeSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide, if_false, inside, if_true]
    refine ⟨body_safe M P p bounds initial current q ctx (ha q) (hb q), ?_⟩
    rw [(bodyExecution M P p initial current q ctx).property]
    exact range_safe M P p bounds initial _ (q+1)
      (bodyInvariant M P p initial current q ctx) ha hb
  · rw [Safety.rangeSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide, if_false, inside]
termination_by steps p-q
decreasing_by omega

theorem source_safe (M : Algebra α) (P : Primitives α) (p : Parameters)
    (bounds : RegionName → Nat) (initial : State α) (hBK : 0<p.blockInner)
    (ha : ∀ q (i : AIndex p), row p initial i.1<p.rows →
      i.2.1.val< inner p-q*p.blockInner → aOffset p initial q i<bounds p.input)
    (hb : ∀ q (i : BIndex p), i.1.val< inner p-q*p.blockInner →
      column p initial i.2.1<p.columns → bOffset p initial q i<bounds p.weight)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds p.output) :
    FP.Structural.TraceSafe (backend M P) bounds (source p) initial := by
  unfold FP.Structural.TraceSafe
  rw [source_decomposition, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨prelude_safe M P p bounds initial, ?_⟩
  intro start ran
  have equal : start=(preludeExecution M P p initial).val :=
    Option.some.inj (ran.symm.trans (preludeExecution M P p initial).property)
  subst start
  obtain ⟨middle, loop, ctx⟩ := loopExecution M P p initial _ 0 (by omega)
    (preludeInvariant M P p initial)
  rw [List.singleton_append, Safety.traceSafeList_unfold]
  refine ⟨?_, ?_⟩
  · have safe := range_safe M P p bounds initial _ 0
      (preludeInvariant M P p initial) ha hb
    simpa [Safety.stepSafe_unfold,loopStmt,source,typed,ComputeKernel.surfaceBody,
      Safety.safeExpr,Safety.safeCompute,Safety.safeOp,Safety.safeOp_unfold,
      evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,integralFloorDiv,
      ComputeDType.eraseDType,Option.bind,body,steps] using safe
  · rw [dynamicLoop M P p _ hBK,loop]
    exact epilogue_safe M P p bounds initial middle ctx hc

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
#axiomsClean range_safe
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.KernelSafety

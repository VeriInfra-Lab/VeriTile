import bench.tritonbench_g.matmul_kernel.Proofs.KernelInputs
import VeriTile.Triton.Float.StructuralSafety

/-! Access bounds follow the actual typed source and its existing loop context.
The statement walks use the shared Structural evaluator and introduce no
floating or hardware-conformance premise. -/
namespace VeriTile.Bench.Repairs.MatmulKernel.KernelSafety
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem prelude_safe (M : Algebra α) (D : DotAcc α)
    (p : Config) (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (backend M D) bounds (prelude p) s := by
  simp [prelude,KernelSource.source,KernelSource.typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod]

theorem body_safe (M : Algebra α) (D : DotAcc α) (p : Config)
    (bounds : RegionName → Nat) (initial current : State α)
    (k : Nat) (a : CIndex p → α) (ctx : Context p initial k current)
    (acc : HasAcc p a current)
    (ha : ∀ i : AIndex p, i.2.1.val+k*p.BK<p.K → aOffset p initial k i<bounds "A")
    (hb : ∀ i : BIndex p, i.1.val+k*p.BK<p.K → bOffset p initial k i<bounds "B") :
    Safety.traceSafeList (backend M D) bounds (body p)
      (current.setReg "k" .nat [] (fun _ => k)) := by
  simp only [HasAcc] at acc
  simp [body,KernelSource.source,KernelSource.typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.offk,ctx.ap,ctx.bp,acc,backend,ha,hb]
  exact ⟨fun i j rest => ha (i,j,rest), fun i j rest => hb (i,j,rest)⟩

theorem epilogue_safe (M : Algebra α) (D : DotAcc α) (p : Config)
    (bounds : RegionName → Nat) (initial current : State α) (k : Nat)
    (ctx : Context p initial k current)
    (acc : HasAcc p (accumulation M D p initial) current)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds "C") :
    Safety.traceSafeList (backend M D) bounds (epilogue p) current := by
  simp only [HasAcc] at acc
  simp [epilogue,KernelSource.source,KernelSource.typed,ComputeKernel.surfaceBody,
    Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,ctx.memory,ctx.pm,ctx.pn,acc,backend]

  intro i j hr hn
  simpa [cOffset,row,col,BM,BN] using hc (i,j,PUnit.unit) ⟨hr,hn⟩

/-- Loop safety follows exactly the source runner's successor context. -/
theorem range_safe (M : Algebra α) (D : DotAcc α) (p : Config)
    (bounds : RegionName → Nat) (initial current : State α)
    (k : Nat) (a : CIndex p → α) (ctx : Context p initial k current)
    (acc : HasAcc p a current)
    (ha : ∀ q (i : AIndex p), i.2.1.val+q*p.BK<p.K → aOffset p initial q i<bounds "A")
    (hb : ∀ q (i : BIndex p), i.1.val+q*p.BK<p.K → bOffset p initial q i<bounds "B") :
    Safety.rangeSafe (backend M D) bounds "k" k p.steps 1 (body p) current := by
  by_cases inside : k<p.steps
  · rw [Safety.rangeSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide, if_false, inside, if_true]
    refine ⟨body_safe M D p bounds initial current k a ctx acc (ha k) (hb k), ?_⟩
    rw [(bodyExecution M D p initial current k a ctx acc).property]
    exact range_safe M D p bounds initial _ (k+1) (update M D p initial k a)
      (bodyContext M D p initial current k a ctx acc).1
      (bodyContext M D p initial current k a ctx acc).2 ha hb
  · rw [Safety.rangeSafe_unfold]
    simp only [show ¬(1:Nat)=0 by decide, if_false, inside]
termination_by p.steps-k
decreasing_by omega

/-- Source access safety for every actual load/store, including all iterations
and masked tails. The hypotheses are only active input/output window bounds. -/
theorem source_safe (M : Algebra α) (D : DotAcc α) (p : Config)
    (bounds : RegionName → Nat) (initial : State α)
    (ha : ∀ q (i : AIndex p), i.2.1.val+q*p.BK<p.K → aOffset p initial q i<bounds "A")
    (hb : ∀ q (i : BIndex p), i.1.val+q*p.BK<p.K → bOffset p initial q i<bounds "B")
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i<bounds "C") :
    FP.Structural.TraceSafe (backend M D) bounds (KernelSource.source p) initial := by
  unfold FP.Structural.TraceSafe
  rw [source_decomposition, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨prelude_safe M D p bounds initial, ?_⟩
  intro start ran
  have equal : start=(preludeExecution M D p initial).val :=
    Option.some.inj (ran.symm.trans (preludeExecution M D p initial).property)
  subst start
  obtain ⟨middle, loop, k, ctx, acc⟩ := loopExecution M D p initial _
    (preludeContext M D p initial).1 (preludeContext M D p initial).2
  rw [List.singleton_append, Safety.traceSafeList_unfold]
  refine ⟨?_, ?_⟩
  · simpa only [Safety.stepSafe_unfold] using range_safe M D p bounds initial _ 0 _
      (preludeContext M D p initial).1 (preludeContext M D p initial).2 ha hb
  · simp only [FP.Structural.step]
    change (match FP.Structural.range (backend M D) "k" 0 p.steps 1 (body p)
        (preludeExecution M D p initial).val with
      | some after => Safety.traceSafeList (backend M D) bounds (epilogue p) after
      | none => True)
    rw [loop]
    exact epilogue_safe M D p bounds initial middle k ctx acc hc

#axiomsClean prelude_safe
#axiomsClean body_safe
#axiomsClean epilogue_safe
#axiomsClean range_safe
#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatmulKernel.KernelSafety

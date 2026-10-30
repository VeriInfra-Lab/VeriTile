import bench.tritonbench_g.matmul_leakyrelu.Proofs.FPFusedComplete
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourceSafety
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.MatmulLeakyrelu.FPFused
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem append_safe (m : Algebra α) (bounds : RegionName → Nat)
    (a b : List ComputeStmt) (s t : State α)
    (safe : Safety.traceSafeList m bounds a s) (ran : run m a s=some t)
    (next : Safety.traceSafeList m bounds b t) :
    Safety.traceSafeList m bounds (a++b) s := by
  apply (Safety.traceSafeList_append m bounds a b s).mpr
  refine ⟨safe,?_⟩
  intro u hu
  have eq : u=t := Option.some.inj (hu.symm.trans ran)
  simpa only [eq] using next

theorem prelude_safe (M : Algebra α) (D : DotAcc α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat) :
    Safety.traceSafeList (backend M D) bounds (prelude p .single) s := by
  simp [prelude,source,typed,dtype,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,run,step,evalExpr,
    evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod]

theorem body_safe (M : Algebra α) (D : DotAcc α) (p : Parameters)
    (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current)
    (bounds : RegionName → Nat)
    (hA : ∀ i : AIndex p,i.2.1.val<p.K-k*BK p → aOffset p initial k i<bounds p.A)
    (hB : ∀ i : BIndex p,i.1.val<p.K-k*BK p → bOffset p initial k i<bounds p.B) :
    Safety.traceSafeList (backend M D) bounds (body p .single)
      (current.setReg "k" .nat [] (fun _ => k)) := by
  simp only [HasAcc] at ha
  simp [body,source,typed,dtype,ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
    Safety.maskActive_unfold,Safety.precisionContext,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericEq,
    ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
    FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
    ctx.offk,ctx.ap,ctx.bp,ha,backend,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    TileShape.insertAxis,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex]
  constructor
  · intro i j tail active
    exact hA (i,j,tail) active
  · intro i j tail active
    exact hB (i,j,tail) active

theorem range_safe (M : Algebra α) (D : DotAcc α) (p : Parameters)
    (initial current : State α) (k : Nat) (a : CIndex p → α)
    (ctx : Context p initial k current) (ha : HasAcc p a current)
    (bounds : RegionName → Nat)
    (hA : ∀ k (i : AIndex p),i.2.1.val<p.K-k*BK p → aOffset p initial k i<bounds p.A)
    (hB : ∀ k (i : BIndex p),i.1.val<p.K-k*BK p → bOffset p initial k i<bounds p.B) :
    Safety.rangeSafe (backend M D) bounds "k" k (trips p) 1 (body p .single) current := by
  rw [Safety.rangeSafe_unfold]
  simp only [Nat.one_ne_zero,if_false]
  split
  · let next := bodyExecution M D p .single initial current k a ctx ha
    refine ⟨body_safe M D p initial current k a ctx ha bounds (hA k) (hB k),?_⟩
    rw [next.property]
    exact range_safe M D p initial next.val (k+1) (update M D p .single initial k a)
      (bodyContext M D p .single initial current k a ctx ha).1
      (bodyContext M D p .single initial current k a ctx ha).2 bounds hA hB
  · trivial
termination_by trips p-k
decreasing_by omega

theorem epilogue_safe (M : Algebra α) (D : DotAcc α) (p : Parameters)
    (initial current : State α) (k : Nat)
    (ctx : Context p initial k current) (ha : HasAcc p (accumulation M D p .single initial) current)
    (bounds : RegionName → Nat)
    (hC : ∀ i : CIndex p,active p initial i → cOffset p initial i<bounds p.C) :
    Safety.traceSafeList (backend M D) bounds (epilogue p .single) current := by
  simp only [HasAcc] at ha
  cases he : p.ACTIVATION <;>
    simp [epilogue,source,typed,dtype,ComputeKernel.surfaceBody,
      Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeAlg_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,
      Safety.maskActive_unfold,Safety.precisionContext,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,numeric,bop,natLt,numericEq,numericLe,
      ComputeDType.eraseDType,ofFloat,toFloat,FloatDType.numericDType,
      FloatDType.toTileDType,Region.cast,Option.bind,integralFloorDiv,integralMod,
      ctx.pm,ctx.pn,ha,backend,he,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
      TileShape.insertAxis,TileShape.dropInsertedIndex,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals simpa [active,cOffset,row,col,Nat.mul_comm] using hC

theorem source_safe (M : Algebra α) (D : DotAcc α) (p : Parameters)
    (initial : State α) (bounds : RegionName → Nat)
    (hA : ∀ k (i : AIndex p),i.2.1.val<p.K-k*BK p → aOffset p initial k i<bounds p.A)
    (hB : ∀ k (i : BIndex p),i.1.val<p.K-k*BK p → bOffset p initial k i<bounds p.B)
    (hC : ∀ i : CIndex p,active p initial i → cOffset p initial i<bounds p.C) :
    TraceSafe (backend M D) bounds (source p .single) initial := by
  let start := preludeExecution M D p .single initial
  have ctx := (preludeContext M D p .single initial).1
  have ha := (preludeContext M D p .single initial).2
  obtain ⟨mid,hm,k,hc,acc⟩ := loopExecution M D p .single initial start.val ctx ha
  change Safety.traceSafeList _ _ (source p .single).surfaceBody initial
  rw [source_decomposition]
  simp only [List.append_assoc]
  apply append_safe _ _ _ _ initial start.val (prelude_safe M D p initial bounds) start.property
  apply append_safe _ _ _ _ start.val mid
  · apply Safety.traceSafeList_cons_intro
    · simpa [Safety.stepSafe_unfold,Safety.safeOp_unfold,tripExpr,trips,
        evalOp_unfold,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex] using
        range_safe M D p initial start.val 0 _ ctx ha bounds hA hB
    · intro t _
      exact Safety.traceSafeList_nil _ _ _
  · simpa [run,step,tripExpr,trips,evalOp_unfold,numeric,bop,
      Broadcast.leftIndex,Broadcast.rightIndex] using hm
  · exact epilogue_safe M D p initial mid k hc acc bounds hC

#axiomsClean source_safe
end
end VeriTile.Bench.Repairs.MatmulLeakyrelu.KernelOnly.SourceSafety

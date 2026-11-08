import bench.tritonbench_g.matmul_triton_autotune.Proofs.Execution
import VeriTile.Triton.Float.StructuralSafety

/-! Access safety follows the actual selected quotation, its live dot carry
and the existing source loop context. Only active memory-window bounds enter;
no numerical or compiler-conformance assumption supplies a program proof. -/
namespace VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {A : Type} [Inhabited A]

theorem prelude_safe (M : Algebra A) (D : DotAcc A) (p : Parameters)
    (bounds : RegionName → Nat) (s : State A) :
    Safety.traceSafeList (backend M D) bounds (prelude p) s := by
  simp [prelude, kernel, typed, Pasted.candidateKernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric, bop, natLt, numericLt, numericEq,
    ComputeDType.eraseDType, ofFloat, toFloat, FloatDType.numericDType,
    FloatDType.toTileDType, Region.cast, Option.bind, integralFloorDiv, integralMod]

theorem body_safe (M : Algebra A) (D : DotAcc A) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State A) (k : Nat) (a : CIndex p → A)
    (ctx : Context p initial current) (acc : HasAcc p a current)
    (ha : ∀ i : AIndex p, k * BK p + i.2.1.val < p.K →
      aRow p initial i.1 * p.stride_am + (k * BK p + i.2.1.val) * p.stride_ak < bounds p.A)
    (hb : ∀ i : BIndex p, k * BK p + i.1.val < p.K →
      (k * BK p + i.1.val) * p.stride_bk + bCol p initial i.2.1 * p.stride_bn < bounds p.B) :
    Safety.traceSafeList (backend M D) bounds (body p)
      (current.setReg "k" .int [] (fun _ => (k : Int))) := by
  simp only [HasAcc] at acc
  simp [body, kernel, typed, Pasted.candidateKernel, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    numeric, bop, natLt, numericLt, numericEq, ComputeDType.eraseDType, ofFloat, toFloat,
    FloatDType.numericDType, FloatDType.toTileDType, Region.cast, Option.bind,
    ctx.memory, ctx.offk, ctx.am, ctx.bn, acc, backend, State.setReg]
  constructor
  · intro i j active
    have hn : k*BK p+j.val<p.K := by simpa only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt] using active
    simpa only [←Int.natCast_add,←Int.natCast_mul,Int.ofNat_lt,hn,if_true,Int.toNat_natCast] using ha (i,j,PUnit.unit) hn
  · intro i j active
    have hn : k*BK p+i.val<p.K := by simpa only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt] using active
    simpa only [←Int.natCast_add,←Int.natCast_mul,Int.ofNat_lt,hn,if_true,Int.toNat_natCast] using hb (i,j,PUnit.unit) hn

theorem epilogue_safe (M : Algebra A) (D : DotAcc A) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State A)
    (ctx : Context p initial current)
    (acc : HasAcc p (accumulation M D p .single initial) current)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i < bounds p.C) :
    Safety.traceSafeList (backend M D) bounds (epilogue p) current := by
  simp only [HasAcc] at acc
  cases enabled : p.ACTIVATION
  all_goals simp [epilogue, kernel, typed, Pasted.candidateKernel, enabled, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, evalExpr, evalComputeOp, evalOp_unfold, evalOffsets,
    store, numeric, bop, natLt, numericLt, numericEq, numericLe, ComputeDType.eraseDType,
    ofFloat, toFloat, FloatDType.numericDType, FloatDType.toTileDType, Region.cast,
    Option.bind, ctx.memory, ctx.pm, ctx.pn, acc, backend, State.setReg]
  all_goals
    intro i j hr hn
    have hr' : pidM p initial*BM p+i.val<p.M := by simpa only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt] using hr
    have hn' : pidN p initial*BN p+j.val<p.N := by simpa only [←Int.natCast_mul,←Int.natCast_add,Int.ofNat_lt] using hn
    have bound := hc (i,j,PUnit.unit) ⟨hr',hn'⟩
    simp only [cOffset,row,col] at bound
    simpa only [←Int.natCast_add,←Int.natCast_mul,Int.ofNat_lt,hr',hn',if_true,Int.toNat_natCast] using bound

theorem range_safe (M : Algebra A) (D : DotAcc A) (p : Parameters)
    (bounds : RegionName → Nat) (initial current : State A) (k : Nat) (a : CIndex p → A)
    (ctx : Context p initial current) (acc : HasAcc p a current)
    (ha : ∀ q (i : AIndex p), q * BK p + i.2.1.val < p.K →
      aRow p initial i.1 * p.stride_am + (q * BK p + i.2.1.val) * p.stride_ak < bounds p.A)
    (hb : ∀ q (i : BIndex p), q * BK p + i.1.val < p.K →
      (q * BK p + i.1.val) * p.stride_bk + bCol p initial i.2.1 * p.stride_bn < bounds p.B) :
    Safety.rangeIntSafe (backend M D) bounds "k" (k : Int) (p.numKBlocks : Int) 1 (body p) current := by
  by_cases inside : k < p.numKBlocks
  · have hi : (k : Int)<(p.numKBlocks : Int) := by exact_mod_cast inside
    rw [Safety.rangeIntSafe_unfold]
    simp only [show ¬(1 : Nat) = 0 by decide, if_false, hi, if_true]
    refine ⟨body_safe M D p bounds initial current k a ctx acc (ha k) (hb k), ?_⟩
    rw [(bodyExecution M D p initial current k a ctx acc).property]
    simpa only [Int.natCast_add,Int.natCast_one] using range_safe M D p bounds initial _ (k + 1) (update M D p .single initial k a)
      (bodyContext M D p initial current k a ctx acc).1
      (bodyContext M D p initial current k a ctx acc).2 ha hb
  · have hi : ¬(k : Int)<(p.numKBlocks : Int) := by exact_mod_cast inside
    rw [Safety.rangeIntSafe_unfold]
    simp only [show ¬(1 : Nat) = 0 by decide, if_false, hi]
termination_by p.numKBlocks - k
decreasing_by omega

theorem loop_step_safe (M : Algebra A) (D : DotAcc A) (p : Parameters)
    (bounds : RegionName→Nat) (s : State A) :
    Safety.stepSafe (backend M D) bounds (loopStatement p) s=
      Safety.rangeIntSafe (backend M D) bounds "k" 0 (p.numKBlocks : Int) 1 (body p) s := by
  rw [←FPValue.loopStop_value]
  simp [loopStatement,kernel,typed,Pasted.candidateKernel,ComputeKernel.surfaceBody,
    Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,Safety.safeOp.eq_def,
    evalExpr,evalComputeOp,evalOp_unfold,numeric,numericLt,integralFloorDiv,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,FPValue.loopStop]
  rfl

theorem source_safe (M : Algebra A) (D : DotAcc A) (p : Parameters)
    (bounds : RegionName → Nat) (initial : State A) (geo : FPValue.Geometry p initial)
    (ha : ∀ q (i : AIndex p), q * BK p + i.2.1.val < p.K →
      aRow p initial i.1 * p.stride_am + (q * BK p + i.2.1.val) * p.stride_ak < bounds p.A)
    (hb : ∀ q (i : BIndex p), q * BK p + i.1.val < p.K →
      (q * BK p + i.1.val) * p.stride_bk + bCol p initial i.2.1 * p.stride_bn < bounds p.B)
    (hc : ∀ i : CIndex p, active p initial i → cOffset p initial i < bounds p.C) :
    FP.Structural.TraceSafe (backend M D) bounds (kernel p) initial := by
  unfold FP.Structural.TraceSafe
  rw [decomposition, List.append_assoc, Safety.traceSafeList_append]
  refine ⟨prelude_safe M D p bounds initial, ?_⟩
  intro start ran
  have equal : start = (preludeExecution M D p initial).val :=
    Option.some.inj (ran.symm.trans (preludeExecution M D p initial).property)
  subst start
  obtain ⟨middle, loop, ctx, acc⟩ := loopExecution M D p initial _
    (preludeContext M D p initial geo).1 (preludeContext M D p initial geo).2
  rw [List.singleton_append, Safety.traceSafeList_unfold]
  refine ⟨?_, ?_⟩
  · rw [loop_step_safe]
    exact range_safe M D p bounds initial _ 0 _
      (preludeContext M D p initial geo).1 (preludeContext M D p initial geo).2 ha hb
  · have hrun : run (backend M D) [loopStatement p] (preludeExecution M D p initial).val=some middle := by
      change run (backend M D) [FPValue.loopStatement p .single] _=some middle
      rw [FPValue.loop_run,FPValue.loopStop_value]
      exact loop
    have hstep : step (backend M D) (loopStatement p) (preludeExecution M D p initial).val=some middle := by
      cases hs : step (backend M D) (loopStatement p) (preludeExecution M D p initial).val <;>
        simpa [run,hs] using hrun
    rw [hstep]
    exact epilogue_safe M D p bounds initial middle ctx acc hc

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTritonAutotune.Repair.Source

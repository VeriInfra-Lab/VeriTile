import bench.tritonbench_g.int_scaled_matmul.IO
import bench.tritonbench_g.int_scaled_matmul.Proofs.RealGrid
import bench.tritonbench_g.int_scaled_matmul.Proofs.PureSafety

namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealScaledSafety
open VeriTile.Triton FP.Structural Revision001 RealScaled
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
attribute [local irreducible] FP.mul FP.round FP.IntegerConversion.toSigned FP.IntegerConversion.totalSigned

private theorem singleton_safe (M : Algebra ℝ) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State ℝ) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp

theorem prelude_safe (rest : Algebra ℝ)
    (p : Parameters) (bounds : RegionName → Nat) (s : State ℝ) :
    Safety.traceSafeList (backend rest) bounds (prelude p) s := by
  apply PureSafety.trace
  intro st member current
  dsimp only [prelude,Revision001.ScaledWord.prelude,original,Revision001.ScaledWord.original,typedScaled,Revision001.ScaledWord.typedScaled,VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledCandidateKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledCandidateKernel,
    ComputeKernel.surfaceBody,List.take] at member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with (rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl)
  all_goals simp [Safety.stepSafe,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

/-- The actual EVEN_K branch or evaluated K-tail mask controls each input
access. Signed8 conversion changes values, never the source addresses. -/
theorem body_safe (rest : Algebra ℝ)
    (p : Parameters) (bounds : RegionName → Nat) (initial current : State ℝ) (k : Nat)
    (h : Invariant p initial k current)
    (ha : ∀ i : AIndex p,p.even=Bool.true ∨ inputInner p initial i.2.1<p.inner-k*p.blockInner →
      aOffset p initial k i<bounds p.input)
    (hb : ∀ i : BIndex p,p.even=Bool.true ∨ inputInner p initial i.1<p.inner-k*p.blockInner →
      bOffset p initial k i<bounds p.weight) :
    Safety.traceSafeList (backend rest) bounds (body p)
      (current.setReg "__rev_k" .int [] (fun _ => Revision001.ScaledWord.mirror p k)) := by
  have mask_eq (t : Nat) : ((t:Int)<(p.inner:Int)-(k*p.blockInner:Int)) ↔ t<p.inner-k*p.blockInner := by omega
  have tail : ∀ s : State ℝ,Safety.traceSafeList (backend rest) bounds ((body p).drop 2) s := by
    intro s
    apply PureSafety.trace
    intro st member current
    simp [body,Revision001.ScaledWord.body,originalBody, Revision001.ScaledWord.statement,Revision001.ScaledWord.originalBody, Revision001.ScaledWord.statement,original,Revision001.ScaledWord.original,typedScaled,Revision001.ScaledWord.typedScaled,VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
      VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
      VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
      VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledCandidateKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledCandidateKernel,
    ComputeKernel.surfaceBody,
      List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with (rfl | rfl | rfl)
    all_goals simp [Safety.stepSafe,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  rw [← List.take_append_drop 2 (body p),Safety.traceSafeList_append]
  refine ⟨?_,fun s _=>tail s⟩
  cases even : p.even
  all_goals
    simp only [even,Bool.false_eq_true,false_or,true_or,true_implies] at ha hb
    simp [body,Revision001.ScaledWord.body,originalBody, Revision001.ScaledWord.statement,Revision001.ScaledWord.originalBody, Revision001.ScaledWord.statement,original,Revision001.ScaledWord.original,typedScaled,Revision001.ScaledWord.typedScaled,VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledCandidateKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledCandidateKernel,
    ComputeKernel.surfaceBody,singleton_safe,Safety.traceSafeList,Safety.stepSafe,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress,
      Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.step,FP.Structural.run,evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.eval_intToSigned,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
      h.offsk,h.aptrs,h.bptrs,h.acc,even,Revision001.ScaledWord.mirror,mask_eq,zero_sub,neg_sub]
  all_goals first
    | exact ⟨fun i j rest h=>ha (i,j,rest) h,fun i j rest h=>hb (i,j,rest) h⟩
    | exact ⟨fun i j rest=>ha (i,j,rest),fun i j rest=>hb (i,j,rest)⟩

theorem epilogue_safe (rest : Algebra ℝ)
    (p : Parameters) (bounds : RegionName → Nat) (initial current : State ℝ)
    (h : OutputReady p initial current)
    (hs : ∀ i : CIndex p,active p initial i → outputRow p initial i*p.scaleRowStride<bounds p.scale)
    (ho : ∀ i : CIndex p,active p initial i → outputOffset p initial i<bounds p.output) :
    Safety.traceSafeList (backend rest) bounds (epilogue p) current := by
  simp [epilogue,Revision001.ScaledWord.epilogue,original,Revision001.ScaledWord.original,typedScaled,Revision001.ScaledWord.typedScaled,VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.unscaledCandidateKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledKernel,
    VeriTile.Bench.Repairs.IntScaledMatmul.Body.scaledCandidateKernel,
    ComputeKernel.surfaceBody,
    singleton_safe,Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,Safety.precisionContext,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
    h.pidm,h.pidn,h.acc,backend]
  constructor
  · intro r c hr hc
    exact hs (r,c,PUnit.unit) ⟨hr,hc⟩
  · intro r c hr hc
    exact ho (r,c,PUnit.unit) ⟨hr,hc⟩

theorem loop_safe (rest : Algebra ℝ)
    (p : Parameters) (cover : p.inner≤p.steps*p.blockInner) (bounds : RegionName → Nat) (initial current : State ℝ) (k : Nat)
    (hk : k≤p.steps) (h : Invariant p initial k current)
    (ha : ∀ k,k<p.steps → ∀ i : AIndex p,
      p.even=Bool.true ∨ inputInner p initial i.2.1<p.inner-k*p.blockInner →
      aOffset p initial k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,
      p.even=Bool.true ∨ inputInner p initial i.1<p.inner-k*p.blockInner →
      bOffset p initial k i<bounds p.weight) :
    Safety.rangeIntSafe (backend rest) bounds "__rev_k" (Revision001.ScaledWord.mirror p k) 0 p.blockInner (body p) current := by
  have aux : ∀ n k (current : State ℝ),p.steps-k=n → k≤p.steps →
      Invariant p initial k current →
      Safety.rangeIntSafe (backend rest) bounds "__rev_k" (Revision001.ScaledWord.mirror p k) 0 p.blockInner (body p) current := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih=>
      intro k current hn hk h
      by_cases zero : p.blockInner=0
      · rw [Safety.rangeIntSafe_unfold,if_pos zero]
        trivial
      by_cases live : Revision001.ScaledWord.mirror p k<0
      · have active : k*p.blockInner<p.inner := by
          have hx : (k*p.blockInner:Int)<(p.inner:Int) := by
            simpa only [Revision001.ScaledWord.mirror,sub_lt_zero] using live
          exact_mod_cast hx
        have less : k<p.steps := by
          by_contra hn
          have le:=Nat.mul_le_mul_right p.blockInner (Nat.le_of_not_gt hn)
          omega
        let next:=bodyExecution rest p initial current k h
        have nextInvariant:=bodyInvariant rest p initial current k h
        rw [Safety.rangeIntSafe_unfold,if_neg zero,if_pos live,next.property]
        refine ⟨body_safe rest p bounds initial current k h (ha k less) (hb k less),?_⟩
        have step : Revision001.ScaledWord.mirror p k+(p.blockInner:Int)=Revision001.ScaledWord.mirror p (k+1) := by
          simp [Revision001.ScaledWord.mirror,Nat.add_mul]
          ring
        rw [step]
        exact ih (p.steps-(k+1)) (by omega) (k+1) next.val rfl (by omega) nextInvariant
      · rw [Safety.rangeIntSafe_unfold,if_neg zero,if_neg live]
        trivial
  exact aux _ _ _ rfl hk h

theorem complete (rest : Algebra ℝ)
    (p : Parameters) (heven : p.even=Bool.true → p.inner=p.steps*p.blockInner)
    (cover : p.inner≤p.steps*p.blockInner) (bounds : RegionName → Nat) (initial : State ℝ)
    (ha : ∀ k,k<p.steps → ∀ i : AIndex p,
      p.even=Bool.true ∨ inputInner p initial i.2.1<p.inner-k*p.blockInner →
      aOffset p initial k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,
      p.even=Bool.true ∨ inputInner p initial i.1<p.inner-k*p.blockInner →
      bOffset p initial k i<bounds p.weight)
    (hs : ∀ i : CIndex p,active p initial i → outputRow p initial i*p.scaleRowStride<bounds p.scale)
    (ho : ∀ i : CIndex p,active p initial i → outputOffset p initial i<bounds p.output) :
    TraceSafe (backend rest) bounds (source p) initial := by
  let start := preludeExecution rest p initial
  have startInv := preludeInvariant rest p initial
  obtain ⟨mid,executed,midInv⟩ := loopExecution rest p heven cover initial start.val 0 (by omega) startInv
  change Safety.traceSafeList (backend rest) bounds (Kernels.scaled p).surfaceBody initial
  rw [SourcePreservation.scaled]
  change Safety.traceSafeList (backend rest) bounds
    (prelude p ++ [Revision001.ScaledWord.statement p] ++ epilogue p) initial
  rw [Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [Safety.traceSafeList_append]
    refine ⟨prelude_safe rest p bounds initial,?_⟩
    intro s hs
    rw [start.property] at hs
    cases hs
    rw [singleton_safe,Revision001.ScaledWord.statement_safety]
    exact loop_safe rest p cover bounds initial start.val 0 (by omega) startInv ha hb
  · intro final hf
    rw [run_append,start.property] at hf
    simp [Option.bind_some,run,Revision001.ScaledWord.statement_step,executed] at hf
    cases hf
    exact epilogue_safe rest p bounds initial mid midInv hs ho

#axiomsClean complete
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealScaledSafety

import bench.tritonbench_g.int_scaled_matmul.IO
import bench.tritonbench_g.int_scaled_matmul.Proofs.RealGrid
import bench.tritonbench_g.int_scaled_matmul.Proofs.PureSafety

namespace VeriTile.Bench.Repairs.IntScaledMatmul.RealScaledSafety
open VeriTile.Triton FP.Structural Revision001 RealScaled
noncomputable section
set_option maxHeartbeats 300000
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
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,ComputeKernel.surfaceBody,List.take] at member
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
      (current.setReg "j" .nat [] (fun _ => k)) := by
  have tail : ∀ s : State ℝ,Safety.traceSafeList (backend rest) bounds ((body p).drop 2) s := by
    intro s
    apply PureSafety.trace
    intro st member current
    simp [body,Revision001.ScaledWord.body,originalBody,Revision001.ScaledWord.originalBody,original,Revision001.ScaledWord.original,typedScaled,Revision001.ScaledWord.typedScaled,VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
      VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
      VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
      VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,ComputeKernel.surfaceBody,
      List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with (rfl | rfl | rfl)
    all_goals simp [Safety.stepSafe,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]
  rw [← List.take_append_drop 2 (body p),Safety.traceSafeList_append]
  refine ⟨?_,fun s _=>tail s⟩
  cases even : p.even
  all_goals
    simp only [even,Bool.false_eq_true,false_or,true_or,true_implies] at ha hb
    simp [body,Revision001.ScaledWord.body,originalBody,Revision001.ScaledWord.originalBody,original,Revision001.ScaledWord.original,typedScaled,Revision001.ScaledWord.typedScaled,VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,ComputeKernel.surfaceBody,singleton_safe,Safety.traceSafeList,Safety.stepSafe,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress,
      Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      FP.Structural.step,FP.Structural.run,evalExpr,evalComputeOp,evalOp_unfold,FP.Structural.eval_intToSigned,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Broadcast.leftIndex,Broadcast.rightIndex,
      h.offsk,h.aptrs,h.bptrs,h.acc,even]
  all_goals first
    | exact ⟨fun i j rest h=>ha (i,j,rest) h,fun i j rest h=>hb (i,j,rest) h⟩
    | exact ⟨fun i j rest=>ha (i,j,rest),fun i j rest=>hb (i,j,rest)⟩

theorem epilogue_safe (rest : Algebra ℝ)
    (p : Parameters) (bounds : RegionName → Nat) (initial current : State ℝ)
    (h : Invariant p initial p.steps current)
    (hs : ∀ i : CIndex p,active p initial i → outputRow p initial i*p.scaleRowStride<bounds p.scale)
    (ho : ∀ i : CIndex p,active p initial i → outputOffset p initial i<bounds p.output) :
    Safety.traceSafeList (backend rest) bounds (epilogue p) current := by
  simp [epilogue,Revision001.ScaledWord.epilogue,original,Revision001.ScaledWord.original,typedScaled,Revision001.ScaledWord.typedScaled,VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.unscaledSource,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaled,
    VeriTile.Bench.Repairs.IntScaledMatmul.Kernels.scaledSource,ComputeKernel.surfaceBody,
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
    (p : Parameters) (bounds : RegionName → Nat) (initial current : State ℝ) (k : Nat)
    (hk : k≤p.steps) (h : Invariant p initial k current)
    (ha : ∀ k,k<p.steps → ∀ i : AIndex p,
      p.even=Bool.true ∨ inputInner p initial i.2.1<p.inner-k*p.blockInner →
      aOffset p initial k i<bounds p.input)
    (hb : ∀ k,k<p.steps → ∀ i : BIndex p,
      p.even=Bool.true ∨ inputInner p initial i.1<p.inner-k*p.blockInner →
      bOffset p initial k i<bounds p.weight) :
    Safety.rangeSafe (backend rest) bounds "j" k p.steps 1 (body p) current := by
  have aux : ∀ n k (current : State ℝ),p.steps-k=n → k≤p.steps →
      Invariant p initial k current →
      Safety.rangeSafe (backend rest) bounds "j" k p.steps 1 (body p) current := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro k current hn hk h
      by_cases live : k<p.steps
      · let next := bodyExecution rest p initial current k h
        have hi := bodyInvariant rest p initial current k h
        rw [Safety.rangeSafe,if_neg (by decide),if_pos live,next.property]
        exact ⟨body_safe rest p bounds initial current k h (ha k live) (hb k live),
          ih (p.steps-(k+1)) (by omega) (k+1) next.val rfl (by omega) hi⟩
      · rw [Safety.rangeSafe,if_neg (by decide),if_neg live]
        trivial
  exact aux _ _ _ rfl hk h

theorem complete (rest : Algebra ℝ)
    (p : Parameters) (bounds : RegionName → Nat) (initial : State ℝ)
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
  obtain ⟨mid,executed,midInv⟩ := loopExecution rest p initial start.val 0 (by omega) startInv
  change Safety.traceSafeList (backend rest) bounds (Kernels.scaled p).surfaceBody initial
  rw [SourcePreservation.scaled]
  change Safety.traceSafeList (backend rest) bounds
    (prelude p ++ [.forRange "j" 0 p.steps 1 (body p)] ++ epilogue p) initial
  rw [Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · rw [Safety.traceSafeList_append]
    refine ⟨prelude_safe rest p bounds initial,?_⟩
    intro s hs
    rw [start.property] at hs
    cases hs
    rw [singleton_safe,Safety.stepSafe]
    exact loop_safe rest p bounds initial start.val 0 (by omega) startInv ha hb
  · intro final hf
    rw [run_append,start.property] at hf
    simp only [Option.bind_some,run,step,executed] at hf
    cases hf
    exact epilogue_safe rest p bounds initial mid midInv hs ho

#axiomsClean complete
end
end VeriTile.Bench.Repairs.IntScaledMatmul.RealScaledSafety

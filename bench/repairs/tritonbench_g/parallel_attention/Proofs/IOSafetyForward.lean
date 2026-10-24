import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyBounds

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Expand the concrete offset list before checking its members. This avoids
unfolding the recursive checker on an unknown offset expression. -/
theorem safeBlockPtr2 (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName → Nat) (s : State α) (region : RegionName)
    (base : Op .nat []) (parent block strides : List Nat) (a b : Op .nat []) :
    Safety.safeOp M precision bounds s (.makeBlockPtrDynOffsets region base parent block strides [a,b]) ↔
      Safety.safeOp M precision bounds s base ∧ Safety.safeOp M precision bounds s a ∧
        Safety.safeOp M precision bounds s b := by
  rw [Safety.safeOp_unfold]
  simp only [List.mem_cons,List.not_mem_nil,or_false,forall_eq_or_imp,forall_eq]

private theorem pure_list (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun st h=>safe st (by simp [h])) t


private theorem prefix_tail (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((firstPrefix c).drop 10) s := by
  apply pure_list
  intro st hs t
  simp only [firstPrefix,forwardKernel,forwardSource,ComputeKernel.surfaceBody,List.take,List.drop,
    List.mem_cons,List.not_mem_nil,or_false] at hs
  rcases hs with rfl|rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]


private theorem first_body_tail (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((firstBody c.BTL c.BTS c.BK c.BV).drop 2) s := by
  apply pure_list
  intro st hs t
  simp [firstBody,forwardSource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]


private theorem prefix_head (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((firstPrefix c).take 9) s := by
  apply pure_list
  intro st hs t
  simp [firstPrefix,forwardKernel,forwardSource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]


private theorem prefix_pointer (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (s t : State α) (ran : run (withDot M D) ((firstPrefix c).take 9) s = some t) :
    t.regs .blockPtr [c.BTL,c.BK] "p_q" = some (fun _=>
      BlockPtr.mk c.q (s.pids 2*c.s_qk_h) [c.T,c.K] [c.BTL,c.BK] [c.s_qk_t,c.s_qk_d]
        [queryChunk c s,keyChunk c s]) := by
  simp [firstPrefix,forwardKernel,forwardSource,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,
    TileShape.indexToList,ofFloat,toFloat] at ran
  subst t
  simp [State.setReg,queryChunk,keyChunk,integralFloorDiv]


theorem forward_prefix (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s : State α) (hb : ForwardBounds c s bounds) :
    Safety.traceSafeList (withDot M D) bounds (firstPrefix c) s := by
  rw [←List.take_append_drop 9 (firstPrefix c),Safety.traceSafeList_append]
  refine ⟨prefix_head M D c bounds s,?_⟩
  intro t ht
  have hp:=prefix_pointer M D c s t ht
  apply Safety.traceSafeList_cons_intro
  · simp [firstPrefix,forwardKernel,forwardSource,ComputeKernel.surfaceBody,
      Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
      Safety.addressSafe,Safety.maskActive,evalOp_unfold,boundaryZero,hp,
      TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
    intro i j hr hc
    exact hb.query _ _ hr hc
  · intro u _
    exact prefix_tail M D c bounds u


theorem forward_first_body (M : Algebra α) (D : DotPrimitive α) (c : ForwardConfig)
    (bounds : RegionName → Nat) (s t : State α) (start : Nat)
    (acc : ForwardAccumulator α c.BTL c.BV) (h : ForwardReady M c s t start acc)
    (hb : ForwardBounds c s bounds) :
    Safety.traceSafeList (withDot M D) bounds (firstBody c.BTL c.BTS c.BK c.BV) t := by
  rw [←List.take_append_drop 2 (firstBody c.BTL c.BTS c.BK c.BV),Safety.traceSafeList_append]
  refine ⟨?_,fun u _=>first_body_tail M D c bounds u⟩
  simp [firstBody,forwardSource,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,State.setReg,
    h.query,h.output,h.normalizer,h.key,h.value,h.mem,h.undef,h.pids,
    keyPointer,valuePointer,TileShape.indexToList,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,BlockPtr.advance_2d_offsets,ofFloat,toFloat]
  constructor
  · intro i j hr hc
    exact hb.key _ _ hr hc
  · intro i j hr hc
    exact hb.value _ _ hr hc

#axiomsClean forward_prefix
#axiomsClean forward_first_body
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety

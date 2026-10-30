import bench.repairs.tritonbench_g.parallel_attention.Proofs.IOSafetyForward

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem option_truth {β : Type} (v : Option β) :
    (match v with | some _=>True | none=>True) := by cases v <;> trivial

private theorem trace_singleton (M : Algebra α) (bounds : RegionName → Nat)
    (st : ComputeStmt) (s : State α) :
    Safety.traceSafeList M bounds [st] s ↔ Safety.stepSafe M bounds st s := by
  rw [Safety.traceSafeList_cons]
  simp

private theorem pure_list (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun st h=>safe st (by simp [h])) t

theorem query_prefix (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) (hb : QueryBounds c bounds) :
    Safety.traceSafeList (withDot M D) bounds (queryPrefix c) s := by
  simp [queryPrefix,queryKernel,querySource,ComputeKernel.surfaceBody]
  simp only [trace_singleton,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  simp only [safeBlockPtr2]
  simp [queryPrefix,queryKernel,querySource,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,safeBlockPtr2,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,State.setReg,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,natLt,numericLt]
  exact ⟨fun i j hr hc=>hb.query _ _ hr hc,fun i j hr hc=>hb.outGrad _ _ hr hc,
    fun i ht=>by simpa only [Nat.add_assoc] using hb.normGrad _ ht⟩

private theorem first_tail (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((queryFirstBody c).drop 2) s := by
  apply pure_list
  intro st hs t
  simp [queryFirstBody,queryKernel,querySource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl
  all_goals by_cases hv : c.i_v=0
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_unfold,trace_singleton,evalOp_unfold,numericEq,hv,Option.bind,bop,option_truth]
  all_goals split <;> trivial

private theorem second_tail (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((querySecondBody c).drop 2) s := by
  apply pure_list
  intro st hs t
  simp [querySecondBody,queryKernel,querySource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals by_cases hv : c.i_v=0
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_unfold,trace_singleton,evalOp_unfold,numericEq,hv,Option.bind,bop,option_truth]
  all_goals split <;> trivial

theorem query_first_body (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s t : State α) (start : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : QueryReady M c s t start acc)
    (hb : QueryBounds c bounds) :
    Safety.traceSafeList (withDot M D) bounds (queryFirstBody c) t := by
  rw [←List.take_append_drop 2 (queryFirstBody c),Safety.traceSafeList_append]
  refine ⟨?_,fun u _=>first_tail M D c bounds u⟩
  simp [queryFirstBody,queryKernel,querySource,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,Option.bind,
    ComputeDType.eraseDType,Region.cast,withDot,State.setReg,h.key,h.value,
    queryKeyPointer,queryValuePointer,TileShape.indexToList,
    BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
  exact ⟨fun i j hr hc=>hb.key _ _ hr hc,fun i j hr hc=>hb.value _ _ hr hc⟩

theorem query_second_body (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s t : State α) (start : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : QueryDiagonalReady M c s t start acc)
    (hb : QueryBounds c bounds) :
    Safety.traceSafeList (withDot M D) bounds (querySecondBody c) t := by
  have hf:=query_first_body M D c bounds s t start acc h.toQueryReady hb
  rw [←List.take_append_drop 2 (queryFirstBody c),Safety.traceSafeList_append] at hf
  rw [←List.take_append_drop 2 (querySecondBody c),Safety.traceSafeList_append]
  exact ⟨hf.1,fun u _=>second_tail M D c bounds u⟩

theorem query_middle (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds (queryMiddle c) s := by
  apply pure_list
  intro st hs t
  simp [queryMiddle,queryKernel,querySource,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals try simp only [safeBlockPtr2]
  all_goals simp [Safety.safeOp_unfold]

theorem query_suffix (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s t : State α) (start : Nat)
    (acc : TileIndex [c.BTL,c.BK] → α) (h : QueryReady M c s t start acc)
    (hb : QueryBounds c bounds) :
    Safety.traceSafeList (withDot M D) bounds (querySuffix c) t := by
  simp [querySuffix,queryKernel,querySource,ComputeKernel.surfaceBody]
  simp only [trace_singleton,Safety.traceSafeList_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  simp only [safeBlockPtr2]
  simp [querySuffix,queryKernel,querySource,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,safeBlockPtr2,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,State.setReg,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,option_truth]
  refine ⟨hb.queryGrad,?_⟩
  split <;> trivial

#axiomsClean query_prefix
#axiomsClean query_first_body
#axiomsClean query_second_body
#axiomsClean query_middle
#axiomsClean query_suffix
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety

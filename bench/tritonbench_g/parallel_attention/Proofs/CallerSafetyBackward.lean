import bench.tritonbench_g.parallel_attention.Proofs.CallerBackwardWhole
import bench.tritonbench_g.parallel_attention.Proofs.IOSafetyBackwardWhole

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller.SafetyProof
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.ParallelAttentionKernel.IOSafety
noncomputable section
set_option maxHeartbeats 800000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

private theorem option_truth {β : Type} (v : Option β) :
    (match v with | some _=>True | none=>True) := by cases v <;> trivial

private theorem pure_list (M : Algebra α) (bounds : RegionName → Nat)
    (code : List ComputeStmt) (safe : ∀ st∈code,∀ s,Safety.stepSafe M bounds st s)
    (s : State α) : Safety.traceSafeList M bounds code s := by
  induction code generalizing s with
  | nil => exact Safety.traceSafeList_nil _ _ _
  | cons st rest ih =>
    apply Safety.traceSafeList_cons_intro (safe st (by simp) s)
    intro t _
    exact ih (fun st h=>safe st (by simp [h])) t

theorem backward_prefix (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) (r : Registers c.toBackwardConfig s) (hb : BackwardBounds c.toBackwardConfig bounds) :
    Safety.traceSafeList (withDot M D) bounds (backwardPrefix c) s := by
  simp [backwardPrefix,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody]
  simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  simp only [safeBlockPtr2]
  simp [Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,Option.bind,
    r.head,r.chunk,r.keyBlock,r.valueBlock,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,ofFloat,toFloat]
  exact ⟨fun i j hr hc=>hb.key _ _ hr hc,fun i j hr hc=>hb.value _ _ hr hc⟩

private theorem first_tail (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((backwardFirstBody c).drop 8) s := by
  apply pure_list
  intro st hs t
  simp [backwardFirstBody,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals by_cases hv : c.i_v=0
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_unfold,evalOp_unfold,numericEq,hv,Option.bind,bop,option_truth]
  all_goals repeat' first | split | trivial

private theorem second_tail (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds ((backwardSecondBody c).drop 7) s := by
  apply pure_list
  intro st hs t
  simp [backwardSecondBody,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  all_goals by_cases hv : c.i_v=0
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_unfold,evalOp_unfold,numericEq,hv,Option.bind,bop,option_truth]
  all_goals repeat' first | split | trivial

theorem backward_first_body (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (t : State α) (start : Nat)
    (hj : t.regs .int [] "__rev_i"=some (fun _=>-(start : Int))) (r : Registers c.toBackwardConfig t) (hb : BackwardBounds c.toBackwardConfig bounds) :
    Safety.traceSafeList (withDot M D) bounds (backwardFirstBody c) t := by
  rw [←List.take_append_drop 8 (backwardFirstBody c),Safety.traceSafeList_append]
  refine ⟨?_,fun u _=>first_tail M D c bounds u⟩
  simp [backwardFirstBody,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody]
  simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  simp only [safeBlockPtr2]
  simp [Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,Option.bind,
    r.head,r.chunk,r.keyBlock,r.valueBlock,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,hj,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,natLt,numericLt,signed_lane_lt,signed_lane_address]
  exact ⟨fun i k hr hc=>hb.query _ _ hr hc,fun i k hr hc=>hb.outGrad _ _ hr hc,
    fun i ht=>by simpa only [Nat.add_assoc] using hb.normGrad _ ht⟩

theorem backward_second_body (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (t : State α) (start : Nat)
    (hi : t.regs .nat [] "i"=some (fun _=>start)) (r : Registers c.toBackwardConfig t) (hb : BackwardBounds c.toBackwardConfig bounds) :
    Safety.traceSafeList (withDot M D) bounds (backwardSecondBody c) t := by
  rw [←List.take_append_drop 7 (backwardSecondBody c),Safety.traceSafeList_append]
  refine ⟨?_,fun u _=>second_tail M D c bounds u⟩
  simp [backwardSecondBody,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody]
  simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  simp only [safeBlockPtr2]
  simp [Safety.safeOp_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,Option.bind,
    r.head,r.chunk,r.keyBlock,r.valueBlock,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,hi,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,natLt,numericLt,signed_lane_lt,signed_lane_address]
  exact ⟨fun i k hr hc=>hb.query _ _ hr hc,fun i k hr hc=>hb.outGrad _ _ hr hc,
    fun i ht=>by simpa only [Nat.add_assoc] using hb.normGrad _ ht⟩

theorem backward_middle (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (withDot M D) bounds (backwardMiddle c) s := by
  apply pure_list
  intro st hs t
  simp [backwardMiddle,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody] at hs
  rcases hs with rfl|rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem backward_suffix (M : Algebra α) (D : DotPrimitive α) (c : QueryConfig)
    (bounds : RegionName → Nat) (s t : State α)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV) (h : BackwardReady M c s t acc)
    (hb : BackwardBounds c.toBackwardConfig bounds) :
    Safety.traceSafeList (withDot M D) bounds (backwardSuffix c) t := by
  simp [backwardSuffix,kernel,shellSource,backwardPastedKernel,ComputeKernel.surfaceBody]
  simp only [Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  simp only [safeBlockPtr2]
  simp [Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,
    h.head,h.chunk,h.keyBlock,h.valueBlock,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,store,
    h.keyGrad,h.valueGrad,ofFloat,toFloat,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
  exact ⟨fun i j hr hc=>hb.keyGrad i j hr hc,fun i j hr hc=>hb.valueGrad i j hr hc⟩

#axiomsClean backward_prefix
#axiomsClean backward_first_body
#axiomsClean backward_second_body
#axiomsClean backward_middle
#axiomsClean backward_suffix
end
end VeriTile.Bench.Repairs.ParallelAttentionKernel.Caller.SafetyProof

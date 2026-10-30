import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelForwardWhole
import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBounds

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicAccess
open VeriTile.Triton FP.Structural FPStages
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (DotPrimitive withDot ForwardConfig queryChunk keyChunk valueChunk)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure ForwardBounds (bounds : RegionName → Nat) (c : ForwardConfig)
    (s : State α) : Prop where
  query : MatrixBound bounds c.q (s.pids 2*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  key : MatrixBound bounds c.k (s.pids 2*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  value : MatrixBound bounds c.v (s.pids 2*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  output : ∀ i : TileIndex [c.BTL,c.BV],
    queryChunk c s+i.1.val<c.T → valueChunk c s+i.2.1.val<c.V →
    (s.pids 2+c.B*c.H*(s.pids 0/((c.V+c.BV-1)/c.BV)))*c.s_vo_h +
      (queryChunk c s+i.1.val)*c.s_vo_t+(valueChunk c s+i.2.1.val)*c.s_vo_d < bounds c.o

def queryPointer (c : ForwardConfig) (s : State α) : BlockPtr :=
  .mk c.q (s.pids 2*c.s_qk_h) [c.T,c.K] [c.BTL,c.BK] [c.s_qk_t,c.s_qk_d]
    [queryChunk c s,keyChunk c s]

def preQuery (c : ForwardConfig) : List ComputeStmt := (firstPrefix c).take 13

theorem pre_query_run (M : Algebra α) (D : DotPrimitive α)
    (c : ForwardConfig) (s : State α) :
    ∃ t, run (withDot M D) (preQuery c) s=some t ∧
      t.regs .blockPtr [c.BTL,c.BK] "p_q"=some (fun _ => queryPointer c s) := by
  simp [preQuery,firstPrefix,forwardKernel,forwardSource,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,
    Option.bind,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,
    TileShape.indexToList,ofFloat,toFloat,queryPointer,queryChunk,keyChunk]
  rfl

#axiomsClean pre_query_run

theorem forward_prefix_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s : State α)
    (h : ForwardBounds bounds c s) :
    Safety.traceSafeList (withDot M D) bounds (firstPrefix c) s := by
  have split : firstPrefix c=preQuery c++(firstPrefix c).drop 13 :=
    (List.take_append_drop 13 _).symm
  rw [split,Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · apply traceSafeList_of_steps
    simp [preQuery,firstPrefix,forwardKernel,forwardSource,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    try simp only [safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]
  · intro t ht
    obtain ⟨u,hu,hp⟩ := pre_query_run M D c s
    rw [ht] at hu
    cases hu
    apply Safety.traceSafeList_cons_intro
    · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
        evalOp_unfold,hp,true_and]
      intro ptr he
      have pe := Option.some.inj he
      subst ptr
      intro i _ hi
      have active : queryChunk c s+i.1.val<c.T ∧ keyChunk c s+i.2.1.val<c.K := by
        simpa [queryPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets] using hi
      simpa [queryPointer,TileShape.indexToList,BlockPtr.address_2d_offsets] using
        h.query _ _ active.1 active.2
    · intro u hu
      apply traceSafeList_of_steps
      simp [firstPrefix,forwardKernel,forwardSource,ComputeKernel.surfaceBody]
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
      simp [Safety.safeOp_unfold]

theorem forward_loads_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s t : State α) (start : Nat)
    (hk : t.regs .blockPtr [c.BK,c.BTS] "p_k"=some (fun _ => keyPointer c s start))
    (hv : t.regs .blockPtr [c.BTS,c.BV] "p_v"=some (fun _ => valuePointer c s start))
    (hb : ForwardBounds bounds c s) :
    Safety.traceSafeList (withDot M D) bounds ((firstBody c.BTL c.BTS c.BK c.BV).take 2) t := by
  apply Safety.traceSafeList_cons_intro
  · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,hk,true_and]
    intro ptr he
    have pe := Option.some.inj he
    subst ptr
    intro i _ hi
    have active : keyChunk c s+i.1.val<c.K ∧ start+i.2.1.val<c.T := by
      simpa [keyPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets] using hi
    simpa [keyPointer,TileShape.indexToList,BlockPtr.address_2d_offsets] using
      hb.key _ _ active.1 active.2
  · intro u hu
    have vp : u.regs .blockPtr [c.BTS,c.BV] "p_v"=some (fun _ => valuePointer c s start) := by
      rw [assign_keeps_reg _ _ _ _ _ _ _ "p_v" (by decide) _ _ hu]
      exact hv
    apply Safety.traceSafeList_cons_intro
    · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
        evalOp_unfold,vp,true_and]
      intro ptr he
      have pe := Option.some.inj he
      subst ptr
      intro i _ hi
      have active : start+i.1.val<c.T ∧ valueChunk c s+i.2.1.val<c.V := by
        simpa [valuePointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets] using hi
      simpa [valuePointer,TileShape.indexToList,BlockPtr.address_2d_offsets] using
        hb.value _ _ active.1 active.2
    · intro v hv; simp

theorem forward_first_body_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : ForwardReady M c s t start acc) (hb : ForwardBounds bounds c s) :
    Safety.traceSafeList (withDot M D) bounds (firstBody c.BTL c.BTS c.BK c.BV) t := by
  rw [← List.take_append_drop 2 (firstBody c.BTL c.BTS c.BK c.BV),Safety.traceSafeList_append]
  refine ⟨forward_loads_safe M D bounds c s t start h.key h.value hb,?_⟩
  intro u hu
  apply traceSafeList_of_steps
  simp [firstBody,forwardSource,ComputeKernel.surfaceBody]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  try simp only [safeBlockPtr2,safeAdvance2]
  simp [Safety.safeOp_unfold]

theorem forward_middle_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : ForwardReady M c s t start acc) :
    Safety.traceSafeList (withDot M D) bounds (betweenLoops c) t := by
  apply traceSafeList_of_steps
  simp [betweenLoops,forwardKernel,forwardSource,ComputeKernel.surfaceBody]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  try simp only [safeBlockPtr2,safeAdvance2]
  simp [Safety.safeOp_unfold]

theorem forward_second_body_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : DiagonalReady M c s t start acc) (hb : ForwardBounds bounds c s) :
    Safety.traceSafeList (withDot M D) bounds (secondBody c.BTL c.BTS c.BK c.BV) t := by
  rw [← List.take_append_drop 2 (secondBody c.BTL c.BTS c.BK c.BV),Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · exact forward_loads_safe M D bounds c s t start h.key h.value hb
  · intro u hu
    apply traceSafeList_of_steps
    simp [secondBody,forwardSource,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    try simp only [safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem forward_suffix_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : ForwardConfig) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BV] → α)
    (h : ForwardReady M c s t start acc) (hb : ForwardBounds bounds c s) :
    Safety.traceSafeList (withDot M D) bounds (forwardSuffix c) t := by
  simp [forwardSuffix,forwardKernel,forwardSource,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,
    Option.bind,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,
    h.head,h.chunk,h.keyBlock,h.valueBlock,h.output,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,FP.Structural.store,queryChunk,valueChunk,Nat.add_assoc]
  try simp only [safeBlockPtr2,safeAdvance2]
  simp [Safety.safeOp_unfold]
  intro i j hi hj
  simpa only [queryChunk,valueChunk,Nat.add_assoc] using hb.output (i,j,PUnit.unit) hi hj

#axiomsClean forward_prefix_safe
#axiomsClean forward_first_body_safe
#axiomsClean forward_middle_safe
#axiomsClean forward_second_body_safe
#axiomsClean forward_suffix_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicAccess

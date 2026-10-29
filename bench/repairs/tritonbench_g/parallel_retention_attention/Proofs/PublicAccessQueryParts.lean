import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBounds
import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryWhole

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicQueryAccess
open VeriTile.Triton FP.Structural FPQuery PublicAccess
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig)
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 30000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) : Prop where
  gradient : MatrixBound bounds c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  key : MatrixBound bounds c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  value : MatrixBound bounds c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  output : ∀ i : TileIndex [c.BTL,c.BK],
    c.i_c*c.BTL+i.1.val<c.T → c.i_k*c.BK+i.2.1.val<c.K →
    (c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h +
      (c.i_c*c.BTL+i.1.val)*c.s_qk_t+(c.i_k*c.BK+i.2.1.val)*c.s_qk_d < bounds dq

theorem prefix_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s : State α)
    (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds (prefixCode c dq) s := by
  rw [← List.take_append_drop 2 (prefixCode c dq),Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  ·
    simp [prefixCode,source,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,
      Option.bind,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,
      TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
      ofFloat,toFloat]
    try simp only [safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,evalOffsets,Region.cast,TileShape.indexToList,
      BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
    intro i j hi hj
    exact hb.gradient _ _ hi hj
  · intro t ht
    apply traceSafeList_of_steps
    simp [prefixCode,source,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    try simp only [safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem loads_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (t : State α) (start : Nat)
    (hk : t.regs .blockPtr [c.BTS,c.BK] "p_k"=some (fun _ => keyPointer c start))
    (hv : t.regs .blockPtr [c.BV,c.BTS] "p_v"=some (fun _ => valuePointer c start))
    (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds ((firstBody c).take 2) t := by
  apply Safety.traceSafeList_cons_intro
  · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,hk,true_and]
    intro ptr he
    have pe := Option.some.inj he
    subst ptr
    intro i _ hi
    have active : start+i.1.val<c.T ∧ c.i_k*c.BK+i.2.1.val<c.K := by
      simpa [keyPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets] using hi
    simpa [keyPointer,TileShape.indexToList,BlockPtr.address_2d_offsets] using
      hb.key _ _ active.1 active.2
  · intro u hu
    have vp : u.regs .blockPtr [c.BV,c.BTS] "p_v"=some (fun _ => valuePointer c start) := by
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
      have active : c.i_v*c.BV+i.1.val<c.V ∧ start+i.2.1.val<c.T := by
        simpa [valuePointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets] using hi
      simpa [valuePointer,TileShape.indexToList,BlockPtr.address_2d_offsets] using
        hb.value _ _ active.1 active.2
    · intro v hv; simp

theorem first_body_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds (firstBody c) t := by
  rw [← List.take_append_drop 2 (firstBody c),Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · exact loads_safe M D bounds c dq t start h.key h.value hb
  · intro u hu
    apply traceSafeList_of_steps
    simp [firstBody,source,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    try simp only [safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem middle_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) :
    Safety.traceSafeList (withDot M D) bounds (middle c dq) t := by
  apply traceSafeList_of_steps
  simp [middle,source,ComputeKernel.surfaceBody]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  try simp only [safeBlockPtr2,safeAdvance2]
  simp [Safety.safeOp_unfold]

theorem second_body_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : DiagonalReady M c s t start acc) (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds (secondBody c) t := by
  rw [← List.take_append_drop 2 (secondBody c),Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · exact loads_safe M D bounds c dq t start h.key h.value hb
  · intro u hu
    apply traceSafeList_of_steps
    simp [secondBody,source,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    try simp only [safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem suffix_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds (suffix c dq) t := by
  simp [suffix,source,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,
    Option.bind,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,h.queryGrad,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,FP.Structural.store,Nat.add_assoc]
  try simp only [safeBlockPtr2,safeAdvance2]
  simp [Safety.safeOp_unfold]
  intro i j hi hj
  simpa only [Nat.add_assoc] using hb.output (i,j,PUnit.unit) hi hj

#axiomsClean prefix_safe
#axiomsClean first_body_safe
#axiomsClean middle_safe
#axiomsClean second_body_safe
#axiomsClean suffix_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicQueryAccess

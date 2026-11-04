import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBounds
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelBackwardWhole

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicBackwardAccess
open VeriTile.Triton FP.Structural FPStages PublicAccess
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages
  (DotPrimitive withDot BackwardConfig BackwardAccumulator)
noncomputable section
set_option maxHeartbeats 350000
set_option maxRecDepth 6000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (bounds : RegionName → Nat) (c : BackwardConfig) : Prop where
  indices : BackwardIndexBounds c
  query : MatrixBound bounds c.q (c.i_bh*c.s_qk_h) c.K c.T c.s_qk_d c.s_qk_t
  gradient : MatrixBound bounds c.do_ (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  key : MatrixBound bounds c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  value : MatrixBound bounds c.v (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  keyOutput : ∀ i : TileIndex [c.BTL,c.BK],
    c.i_c*c.BTL+i.1.val<c.T → c.i_k*c.BK+i.2.1.val<c.K →
    (c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h +
      (c.i_c*c.BTL+i.1.val)*c.s_qk_t+(c.i_k*c.BK+i.2.1.val)*c.s_qk_d < bounds c.dk
  valueOutput : ∀ i : TileIndex [c.BTL,c.BV],
    c.i_c*c.BTL+i.1.val<c.T → c.i_v*c.BV+i.2.1.val<c.V →
    (c.i_bh+c.B*c.H*c.i_k)*c.s_vo_h +
      (c.i_c*c.BTL+i.1.val)*c.s_vo_t+(c.i_v*c.BV+i.2.1.val)*c.s_vo_d < bounds c.dv

theorem numerical_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (h : BackwardArgumentReady c s t) (hb : Bounds bounds c) :
    Safety.traceSafeList (withDot M D) bounds (backwardNumericalPrefix c) t := by
  rw [←List.take_append_drop 6 (backwardNumericalPrefix c),Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · simp [backwardNumericalPrefix,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,run,step,evalExpr,evalComputeOp,
      evalOp_unfold,boundaryZero,evalOffsets,Option.bind,ComputeDType.eraseDType,numeric,bop,
      withDot,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,hb.indices.query_cast,hb.indices.key_cast,hb.indices.value_cast,
      Index32.natural_product,TileShape.indexToList,BlockPtr.address_2d_offsets,
      BlockPtr.inBounds_2d_offsets,ofFloat,toFloat]
    try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,evalOffsets,Region.cast,TileShape.indexToList,
      BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
    constructor
    · intro i j hi hj;exact hb.key _ _ hi hj
    · intro i j hi hj;exact hb.value _ _ hi hj
  · intro u hu
    apply traceSafeList_of_steps
    simp [backwardNumericalPrefix,backwardKernel,backwardSource,ComputeKernel.surfaceBody]
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem prefix_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s : State α)
    (hb : Bounds bounds c) :
    Safety.traceSafeList (withDot M D) bounds (backwardPrefix c) s := by
  rw [backward_prefix_split,Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · apply traceSafeList_of_steps
    simp [backwardArgumentPrefix,backwardKernel,backwardSource,ComputeKernel.surfaceBody]
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
  · intro t ht
    obtain ⟨u,hu,args⟩:=backward_arguments_run (withDot M D) c s
    rw [ht] at hu
    cases hu
    exact numerical_safe M D bounds c s t args hb

theorem first_body_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (hb : Bounds bounds c)
    (hj : t.regs .int [] "__rev_i" = some (fun _ => -(start : Int)))
    (timeBound : start<2147483648) :
    Safety.traceSafeList (withDot M D) bounds (backwardFirstBody c) t := by
  rw [← List.take_append_drop 5 (backwardFirstBody c),Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  ·
    simp [backwardFirstBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,
      Option.bind,ComputeDType.eraseDType,numeric,bop,numericEq,Region.cast,withDot,
      h.key,h.value,h.keyGrad,h.valueGrad,h.decayLog,h.decayBlock,h.decayHead,
      h.decayQuery,h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,hb.indices.key_cast,hb.indices.value_cast,Index32.nat_value start timeBound,Index32.natural_product,hj,h.mem,h.undef,h.pids,TileShape.indexToList,
      BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,ofFloat,toFloat,
      natLt,numericLt]
    try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,evalOffsets,Region.cast,TileShape.indexToList,
      BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
    constructor
    · intro i j hi hj; exact hb.query _ _ hi hj
    · intro i j hi hj; exact hb.gradient _ _ hi hj
  · intro t ht
    apply traceSafeList_of_steps
    simp [backwardFirstBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem middle_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) :
    Safety.traceSafeList (withDot M D) bounds (backwardMiddle c) t := by
  apply traceSafeList_of_steps
  simp [backwardMiddle,backwardKernel,backwardSource,ComputeKernel.surfaceBody]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
  simp [Safety.safeOp_unfold]

theorem second_body_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (start : Nat) (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardDiagonalReady M c s t start acc) (hb : Bounds bounds c)
    (hi : t.regs .int [] "i" = some (fun _ => (start : Int)))
    (timeBound : start<2147483648) :
    Safety.traceSafeList (withDot M D) bounds (backwardSecondBody c) t := by
  rw [← List.take_append_drop 4 (backwardSecondBody c),Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  ·
    simp [backwardSecondBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
      Safety.safeCompute_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,
      Option.bind,ComputeDType.eraseDType,numeric,bop,numericEq,Region.cast,withDot,
      h.key,h.value,h.keyGrad,h.valueGrad,h.decayLog,h.decayBlock,h.decayHead,
      h.decayQuery,h.keyDecayed,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,hb.indices.key_cast,hb.indices.value_cast,Index32.nat_value start timeBound,Index32.natural_product,hi,h.mem,h.undef,h.pids,h.queryIndex,h.keyIndex,
      TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
      numericLt,numericLe,natLe,natLt,ofFloat,toFloat]
    try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      evalOp_unfold,evalOffsets,Region.cast,TileShape.indexToList,
      BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets]
    constructor
    · intro i j hi hj; exact hb.query _ _ hi hj
    · intro i j hi hj; exact hb.gradient _ _ hi hj
  · intro t ht
    apply traceSafeList_of_steps
    simp [backwardSecondBody,backwardKernel,backwardSource,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem suffix_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (s t : State α)
    (acc : BackwardAccumulator α c.BTL c.BK c.BV)
    (h : BackwardReady M c s t acc) (hb : Bounds bounds c) :
    Safety.traceSafeList (withDot M D) bounds (backwardSuffix c) t := by
  simp [backwardSuffix,backwardKernel,backwardSource,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,boundaryZero,evalOffsets,
    Option.bind,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,
    h.keyGrad,h.valueGrad,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,hb.indices.query_cast,hb.indices.key_cast,hb.indices.value_cast,Index32.natural_product,TileShape.indexToList,BlockPtr.address_2d_offsets,
    BlockPtr.inBounds_2d_offsets,ofFloat,toFloat,FP.Structural.store]
  try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
  simp [Safety.safeOp_unfold]
  constructor
  · intro i j hi hj; exact hb.keyOutput (i,j,PUnit.unit) hi hj
  · intro i j hi hj; exact hb.valueOutput (i,j,PUnit.unit) hi hj

#axiomsClean prefix_safe
#axiomsClean first_body_safe
#axiomsClean middle_safe
#axiomsClean second_body_safe
#axiomsClean suffix_safe
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicBackwardAccess

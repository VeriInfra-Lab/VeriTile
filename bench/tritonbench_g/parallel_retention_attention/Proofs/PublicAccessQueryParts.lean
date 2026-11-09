import bench.tritonbench_g.parallel_retention_attention.Proofs.PublicAccessBounds
import bench.tritonbench_g.parallel_retention_attention.Proofs.KernelQueryWhole

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.PublicQueryAccess
open VeriTile.Triton FP.Structural FPQuery PublicAccess
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (DotPrimitive withDot BackwardConfig)
noncomputable section
set_option maxHeartbeats 250000
set_option maxRecDepth 6000
set_option maxErrors 8
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

structure Bounds (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) : Prop where
  indices : HelperIndexBounds c
  gradient : MatrixBound bounds c.do_ (c.i_bh*c.s_vo_h) c.T c.V c.s_vo_t c.s_vo_d
  key : MatrixBound bounds c.k (c.i_bh*c.s_qk_h) c.T c.K c.s_qk_t c.s_qk_d
  value : MatrixBound bounds c.v (c.i_bh*c.s_vo_h) c.V c.T c.s_vo_d c.s_vo_t
  output : ∀ i : TileIndex [c.BTL,c.BK],
    c.i_c*c.BTL+i.1.val<c.T → c.i_k*c.BK+i.2.1.val<c.K →
    (c.i_bh+c.B*c.H*c.i_v)*c.s_qk_h +
      (c.i_c*c.BTL+i.1.val)*c.s_qk_t+(c.i_k*c.BK+i.2.1.val)*c.s_qk_d < bounds dq

def gradientPointer (c : BackwardConfig) : BlockPtr :=
  .mk c.do_ (c.i_bh*c.s_vo_h) [c.T,c.V] [c.BTL,c.BV] [c.s_vo_t,c.s_vo_d]
    [c.i_c*c.BTL,c.i_v*c.BV]
def gradientSetup (c : BackwardConfig) (dq : RegionName) := (numericalPrefix c dq).take 1

theorem gradient_setup_run (M : Algebra α) (D : DotPrimitive α)
    (c : BackwardConfig) (dq : RegionName) (s t : State α) (h : ArgumentReady c s t)
    (ib : HelperIndexBounds c) :
    ∃ u,run (withDot M D) (gradientSetup c dq) t=some u ∧
      u.regs .blockPtr [c.BTL,c.BV] "p_do"=some (fun _=>gradientPointer c) := by
  simp [gradientSetup,numericalPrefix,source,ComputeKernel.surfaceBody,run,step,
    evalExpr,evalComputeOp,evalOp_unfold,Option.bind,numeric,bop,evalOffsets,
    h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,ib.query_cast,ib.value_cast,Index32.natural_product,
    gradientPointer,TileShape.indexToList,ofFloat,toFloat]

theorem numerical_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (h : ArgumentReady c s t) (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds (numericalPrefix c dq) t := by
  have split : numericalPrefix c dq=gradientSetup c dq++(numericalPrefix c dq).drop 1:=rfl
  rw [split,Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · apply traceSafeList_of_steps
    simp [gradientSetup,numericalPrefix,source,ComputeKernel.surfaceBody]
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
      safeBlockPtrMetadata2]
    simp [Safety.safeOp_unfold]
  · intro u hu
    obtain ⟨v,hv,pointer⟩:=gradient_setup_run M D c dq s t h hb.indices
    rw [hu] at hv
    cases hv
    apply Safety.traceSafeList_cons_intro
    · simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        Safety.safeOp_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
        evalOp_unfold,pointer,true_and]
      intro ptr he
      have pe:=Option.some.inj he
      subst ptr
      intro i _ hi
      have active : c.i_c*c.BTL+i.1.val<c.T ∧ c.i_v*c.BV+i.2.1.val<c.V := by
        simpa [gradientPointer,TileShape.indexToList,BlockPtr.inBounds_2d_offsets] using hi
      simpa [gradientPointer,TileShape.indexToList,BlockPtr.address_2d_offsets] using
        hb.gradient _ _ active.1 active.2
    · intro v hv
      apply traceSafeList_of_steps
      simp [numericalPrefix,source,ComputeKernel.surfaceBody]
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
        safeBlockPtrMetadata2]
      simp [Safety.safeOp_unfold]

theorem prefix_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s : State α)
    (hb : Bounds bounds c dq) :
    Safety.traceSafeList (withDot M D) bounds (prefixCode c dq) s := by
  rw [prefix_split,Safety.traceSafeList_append]
  refine ⟨?_,?_⟩
  · apply traceSafeList_of_steps
    simp [argumentPrefix,source,ComputeKernel.surfaceBody]
    simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold]
  · intro t ht
    obtain ⟨u,hu,args⟩:=arguments_run (withDot M D) c dq s
    rw [ht] at hu
    cases hu
    exact numerical_safe M D bounds c dq s t args hb

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
    try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
    simp [Safety.safeOp_unfold]

theorem middle_safe (M : Algebra α) (D : DotPrimitive α)
    (bounds : RegionName → Nat) (c : BackwardConfig) (dq : RegionName) (s t : State α)
    (start : Nat) (acc : TileIndex [c.BTL,c.BK] → α)
    (h : Ready M c s t start acc) :
    Safety.traceSafeList (withDot M D) bounds (middle c dq) t := by
  apply traceSafeList_of_steps
  simp [middle,source,ComputeKernel.surfaceBody]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
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
    try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
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
    Option.bind,ComputeDType.eraseDType,numeric,bop,Region.cast,withDot,h.queryGrad,h.argument0,h.argument1,h.argument2,h.argument3,h.argument4,h.argument5,h.argument6,h.argument7,h.argument8,h.argument9,h.argument10,h.argument11,h.argument12,h.argument13,hb.indices.query_cast,hb.indices.key_cast,hb.indices.value_cast,Index32.natural_product,
    TileShape.indexToList,BlockPtr.address_2d_offsets,BlockPtr.inBounds_2d_offsets,
    ofFloat,toFloat,FP.Structural.store,Nat.add_assoc]
  try simp only [safeBlockPtrMetadata2,safeBlockPtr2,safeAdvance2]
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

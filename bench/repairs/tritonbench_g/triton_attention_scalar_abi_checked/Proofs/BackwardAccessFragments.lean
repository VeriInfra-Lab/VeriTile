import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.AccessGeometry
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardBody

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem backward_score_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block query : Nat)
    (mem : Memory α) (acc : BackwardInner α p)
    (inv : BackwardInnerInvariant p initial block query mem acc current) (bounds : RegionName→Nat)
    (queries : ∀ i : BLane p,backwardOffset p initial query i<bounds p.q)
    (maximum : ∀ i : BRow p,backwardRowOffset p initial query i<bounds p.maximum) :
    Safety.traceSafeList (backend rest P) bounds ((backwardInnerBody p).take 6)
      (current.setReg "start_m" .nat [] (fun _=>query*p.blockM)) := by
  simp [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,backend,
    numeric,numericLe,natLe,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,
    inv.rows,inv.cols,inv.key,inv.qPtr,inv.maxPtr,backwardPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets]
  refine ⟨fun i d _=>queries (i,d,PUnit.unit),?_⟩
  intro i
  simpa [backwardRowOffset,Nat.add_assoc] using maximum (i,PUnit.unit)

theorem backward_gradient_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block query : Nat)
    (mem : Memory α) (acc : BackwardInner α p)
    (inv : BackwardInnerInvariant p initial block query mem acc current)
    (rows : current.regs .nat [p.blockM] "offs_m_curr"=some (fun i=>query*p.blockM+i.1.val))
    (weights : current.regs .real [p.blockM,p.blockM] "p"=
      some (backwardWeights P p initial acc.mem (backwardRead p initial mem p.k block) block query))
    (bounds : RegionName→Nat)
    (gradient : ∀ i : BLane p,backwardOffset p initial query i<bounds p.gradient)
    (delta : ∀ i : BRow p,backwardRowOffset p initial query i<bounds p.delta) :
    Safety.traceSafeList (backend rest P) bounds (((backwardInnerBody p).drop 6).take 6) current := by
  simp [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,backend,
    numeric,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,
    inv.dv,inv.value,inv.doPtr,inv.deltaPtr,rows,weights,backwardPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets]
  refine ⟨fun i d _=>gradient (i,d,PUnit.unit),?_⟩
  intro i
  simpa [backwardRowOffset,Nat.add_assoc] using delta (i,PUnit.unit)

theorem backward_store_safe (rest : Algebra α) (P : Primitives α)
    (p : BackwardParameters) (initial current : State α) (block query : Nat)
    (mem : Memory α) (acc : BackwardInner α p)
    (pointer : current.regs .blockPtr [p.blockM,p.head] "dq_tile_ptr"=
      some (fun _=>backwardPointer p initial p.dq query))
    (key : current.regs p.dtype.toTileDType [p.blockM,p.head] "k"=some (backwardRead p initial mem p.k block))
    (q : current.regs p.dtype.toTileDType [p.blockM,p.head] "q"=some (backwardRead p initial acc.mem p.q query))
    (dk : current.regs .real [p.blockM,p.head] "dk"=some acc.dk)
    (ds : current.regs .real [p.blockM,p.blockM] "ds"=some
      (backwardDS P p initial acc.mem (backwardRead p initial mem p.k block) (backwardRead p initial mem p.v block) block query))
    (bounds : RegionName→Nat)
    (dq : ∀ i : BLane p,backwardOffset p initial query i<bounds p.dq) :
    Safety.traceSafeList (backend rest P) bounds (((backwardInnerBody p).drop 12).take 4) current := by
  simp [backwardInnerBody,backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,store,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,backend,
    numeric,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,
    pointer,key,q,dk,ds,backwardPointer,TileShape.indexToList,
    BlockPtr.inBounds,BlockPtr.address_2d_offsets]
  exact fun i d=>dq (i,d,PUnit.unit)

theorem backward_advance_safe (M : Algebra α) (p : BackwardParameters)
    (s : State α) (bounds : RegionName→Nat) :
    Safety.traceSafeList M bounds ((backwardInnerBody p).drop 16) s := by
  apply AccessSafety.list_safe_of_steps
  intro statement member state
  change statement∈[_,_,_,_] at member
  simp only [List.mem_cons,List.not_mem_nil,or_false] at member
  rcases member with rfl|rfl|rfl|rfl
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

#axiomsClean backward_score_safe
#axiomsClean backward_gradient_safe
#axiomsClean backward_store_safe
#axiomsClean backward_advance_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess

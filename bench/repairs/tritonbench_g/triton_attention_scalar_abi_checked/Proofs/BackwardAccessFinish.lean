import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.AccessGeometry
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPBackwardFinish

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

@[local simp 2000] theorem two_delta_pointer (M : Algebra α) (precision : Option ComputeDType)
    (bounds : RegionName→Nat) (s : State α) (shape : TileShape)
    (ptr : Op .blockPtr shape) (x y : Op .int []) :
    Safety.safeOp M precision bounds s (.advanceBlockPtrDyn ptr [x,y]) ↔
      Safety.safeOp M precision bounds s ptr ∧ Safety.safeOp M precision bounds s x ∧
        Safety.safeOp M precision bounds s y := by
  rw [Safety.safeOp_unfold]
  simp [List.mem_cons]

theorem backward_finish_safe (rest : Algebra α) (P : Primitives α) (p : BackwardParameters)
    (initial current : State α) (block : Nat) (st : BackwardInner α p)
    (inv : BackwardOuterInvariant p initial block st.mem current)
    (dv : current.regs .real [p.blockM,p.head] "dv"=some st.dv)
    (dk : current.regs .real [p.blockM,p.head] "dk"=some st.dk)
    (lo : current.regs .nat [] "lo"=some (fun _=>block*p.blockM))
    (qptr : current.regs .blockPtr [p.blockM,p.head] "q_tile_ptr"=some (fun _=>backwardPointer p initial p.q p.blocks))
    (doptr : current.regs .blockPtr [p.blockM,p.head] "do_tile_ptr"=some (fun _=>backwardPointer p initial p.gradient p.blocks))
    (dqptr : current.regs .blockPtr [p.blockM,p.head] "dq_tile_ptr"=some (fun _=>backwardPointer p initial p.dq p.blocks))
    (bounds : RegionName→Nat)
    (values : ∀ i : BLane p,backwardOffset p initial block i<bounds p.dv)
    (keys : ∀ i : BLane p,backwardOffset p initial block i<bounds p.dk) :
    Safety.traceSafeList (backend rest P) bounds ((backwardOuterBody p).drop 13) current := by
  simp [backwardOuterBody,backwardSource,backwardTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,store,evalExpr,evalComputeOp,evalOp_unfold,evalDeltas,Option.bind,backend,
    numeric,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,maskedScatter_regs,
    inv.keyPtr,inv.valuePtr,inv.dvPtr,inv.dkPtr,dv,dk,lo,qptr,doptr,dqptr,
    backwardPointer_reset,reset_offset,backwardPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets]
  exact ⟨fun i d _=>values (i,d,PUnit.unit),fun i d _=>keys (i,d,PUnit.unit)⟩

#axiomsClean backward_finish_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess

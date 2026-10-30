import bench.tritonbench_g.block_sparse_attn.Proofs.AccessFragments

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem suffix_safe (a : Arithmetic α) (native : Bool) (p : Parameters)
    (initial current : State (Replicas α)) (carry : Carry α) (slot : Int)
    (h : ChannelInvariant a p initial carry (csrColumn p initial slot)
      (scoreAccumulation a native p initial (csrColumn p initial slot) p.blocks) current)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds)
    (lo : first p initial≤slot) (hi : slot<last p initial) :
    Safety.traceSafeList a.model bounds (suffix native p) current := by
  have hv := backed.value slot lo hi
  cases native <;> cases en : p.evenN
  all_goals
    simp [suffix,loopBody,compiledBody,body,source,declared,en,
      scoreFmaNode,denominatorFmaNode,accScaleNode,carryNode,fp,ref1,ref2,same1,same2,causalMask,
      ComputeKernel.surfaceBody,
      AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
      Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,numericLe,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Arithmetic.model,TileShape.axisDim,
      h.scale,h.memory,h.vptrs,h.rows,h.keys,h.length,h.start,h.scores,h.maximum,h.denominator,h.accumulator,
      Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    intro i j unit
    cases unit
    try intro active
    simpa only [Int.natCast_mul] using hv (i,j,PUnit.unit) (by simp_all)

theorem epilogue_safe (a : Arithmetic α) (p : Parameters)
    (initial current : State (Replicas α)) (carry : Carry α)
    (h : Runtime a p initial carry current)
    (bounds : RegionName→Nat) (backed : AccessBounds p initial bounds) :
    Safety.traceSafeList a.model bounds (epilogue p) current := by
  simp [epilogue,source,declared,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Arithmetic.model,
    h.batch,h.head,h.channel,h.rows,h.dimensions,h.length,h.accumulator,
    Broadcast.leftIndex,Broadcast.rightIndex]
  intro i j active
  exact backed.output (i,j,PUnit.unit) active

#axiomsClean suffix_safe
#axiomsClean epilogue_safe
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly.SourceSafety

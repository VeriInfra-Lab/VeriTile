import bench.repairs.tritonbench_g.block_sparse_attn.KernelOnly.Channel

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false

def suffix (native : Bool) (p : Parameters) := (loopBody native p).drop 4

def visitSuffixExecution {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (column : Int)
    (h : ChannelInvariant a p initial carry column
      (scoreAccumulation a native p initial column p.blocks) current) :
    {t : State (Replicas α) // run a.model (suffix native p) current = some t} := by
  cases native <;> cases en : p.evenN
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [suffix,loopBody,compiledBody,body,source,declared,en,
      scoreFmaNode,denominatorFmaNode,accScaleNode,carryNode,fp,ref1,ref2,same1,same2,causalMask,
      ComputeKernel.surfaceBody,run,FP.Structural.step,evalExpr,evalComputeOp,
      evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,numericLe,
      ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
      integralFloorDiv,integralMod,Arithmetic.model,
      TileShape.axisDim,
      h.scale,h.memory,h.vptrs,h.rows,h.keys,h.length,h.start,h.scores,h.maximum,h.denominator,h.accumulator,
      Broadcast.leftIndex,Broadcast.rightIndex]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

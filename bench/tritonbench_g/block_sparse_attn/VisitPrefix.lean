import bench.tritonbench_g.block_sparse_attn.Channel

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

theorem initialRuntime {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (s : State (Replicas α)) (integers : Int32Memory s) :
    Runtime a p s (initialCarry a) (prefixExecution a.model p (bindScale (broadcast (a.argument p.scale)) s) integers).val := by
  refine ⟨prefixIndices a p s integers,?_,?_,?_⟩ <;>
    simp [prefixExecution,bindScale,initialCarry,Arithmetic.model,binary,broadcast]
  funext i
  rfl

def visitPrefixExecution {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (slot : Int) (h : Runtime a p initial carry current)
    (integers : Int32Memory initial) :
    {t : State (Replicas α) // run a.model ((body p).take 3)
      (current.setReg "col_idx_idx" .int [] (fun _ => slot)) = some t} := by
  have cols := integers "cols" (by simp)
  refine ⟨?_,?_⟩
  swap
  simp [body,source,declared,ComputeKernel.surfaceBody,run,FP.Structural.step,
    evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,numeric,bop,natLt,numericLt,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    integralFloorDiv,integralMod,Arithmetic.model,Int32Bits.valuesValid,
    h.scale,h.memory,h.layout,cols,Broadcast.leftIndex,Broadcast.rightIndex]
  rfl

theorem visitPrefixInvariant {α : Type} [Inhabited α] (a : Arithmetic α)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (slot : Int) (h : Runtime a p initial carry current)
    (integers : Int32Memory initial) :
    ChannelInvariant a p initial carry (csrColumn p initial slot)
      (fun _ => broadcast (a.literal 0))
      (visitPrefixExecution a p initial current carry slot h integers).val := by
  refine ⟨⟨⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩,?_,?_,?_⟩,?_,?_⟩
  all_goals simp [visitPrefixExecution,csrColumn,Arithmetic.model,
    h.scale,h.memory,h.length,h.batch,h.head,h.channel,h.rows,h.keys,h.dimensions,
    h.qptrs,h.kptrs,h.vptrs,h.layout,h.first,h.last,h.maximum,h.denominator,h.accumulator,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals rfl

#axiomsClean initialRuntime
#axiomsClean visitPrefixInvariant
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

import bench.tritonbench_g.block_sparse_attn.ChannelExecution

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false

theorem channelInvariant {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (column : Int) (scores : Matrix α) (block : Nat)
    (h : ChannelInvariant a p initial carry column scores current) :
    ChannelInvariant a p initial carry column (scoreStep a native p initial column scores block)
      (channelExecution a native p initial current carry column scores block h).val := by
  rcases p with ⟨rs,cs,ls,scale,qb,qh,qm,qd,kb,kh,kn,kd,vb,vh,vn,vd,ob,oh,om,heads,kvHeads,len,blocks,evenM,evenN⟩
  cases native <;> cases evenM <;> cases evenN
  all_goals
    refine ⟨⟨⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩,?_,?_,?_⟩,?_,?_⟩
    all_goals simp [channelExecution,scoreStep,query,key,Arithmetic.model,
      h.scale,h.memory,h.length,h.batch,h.head,h.channel,h.rows,h.keys,h.dimensions,
      h.qptrs,h.kptrs,h.vptrs,h.layout,h.first,h.last,h.maximum,h.denominator,h.accumulator,h.start,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    funext i
    try delta query key
    try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
    try rfl

#axiomsClean channelInvariant
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

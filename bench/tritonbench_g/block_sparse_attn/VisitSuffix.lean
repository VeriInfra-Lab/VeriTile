import bench.tritonbench_g.block_sparse_attn.VisitSuffixExecution

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false

theorem visitSuffixInvariant {α : Type} [Inhabited α] (a : Arithmetic α) (native : Bool)
    (p : Parameters) (initial current : State (Replicas α))
    (carry : Carry α) (column : Int)
    (h : ChannelInvariant a p initial carry column
      (scoreAccumulation a native p initial column p.blocks) current) :
    Runtime a p initial (update a native p initial column carry)
      (visitSuffixExecution a native p initial current carry column h).val := by
  rcases p with ⟨rs,cs,ls,scale,qb,qh,qm,qd,kb,kh,kn,kd,vb,vh,vn,vd,ob,oh,om,heads,kvHeads,len,blocks,evenM,evenN⟩
  cases native <;> cases evenN
  all_goals
    refine ⟨⟨?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_,?_⟩,?_,?_,?_⟩
    all_goals simp [visitSuffixExecution,update,value,Arithmetic.model,
      h.scale,h.memory,h.length,h.batch,h.head,h.channel,h.rows,h.keys,h.dimensions,
      h.qptrs,h.kptrs,h.vptrs,h.layout,h.first,h.last,
      binary,broadcast,reduction,FP.Structural.bop,
      Broadcast.leftIndex,Broadcast.rightIndex,TileShape.insertAxisIndex]
  all_goals funext i o
  all_goals try delta value
  all_goals try simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    binary,broadcast,ite_apply]
  all_goals rfl

#axiomsClean visitSuffixInvariant
end
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

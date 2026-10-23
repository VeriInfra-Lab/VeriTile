import bench.repairs.tritonbench_g.context_attn_nopad.KernelOnly.Geometry

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section

/-- Every valid logical output has an active lane in the complete source grid. -/
theorem logical_coverage (r : Request) (heap : ByteMemory) (h : Contract r heap)
    (b head t d : Nat) (hb : b<r.call.batches) (hh : head<r.call.q.heads)
    (ht : t<r.call.length b) (hd : d<r.call.q.dims) :
    ∃ idx : GridIndex r.call.grid,∃ i : TileIndex [r.call.layout.BM,r.call.layout.D],
      FPReference.laneActive r.call idx i ∧
      flatIndex r.call idx i=((r.call.start b+t)*r.call.q.heads+head)*r.call.q.dims+d ∧
      FPReference.laneOffset r.call idx i=r.call.output b head t d := by
  let hm := layout_memory r heap h
  obtain ⟨idx,i,batch,headEq,time,dim,on⟩ := Host.covered hm b head t d hb hh ht hd
    (lt_of_lt_of_le ht (le_trans (complete_grid r heap h b hb) (blocks_cover r.call)))
  have active : FPReference.laneActive r.call idx i := (Host.active_iff hm idx i).mp on
  have timeEq : (r.call.launch idx).block*128+i.1.val=t := time
  have flat : flatIndex r.call idx i=((r.call.start b+t)*r.call.q.heads+head)*r.call.q.dims+d := by
    simp only [flatIndex,batch,headEq,timeEq,dim]
  refine ⟨idx,i,active,flat,?_⟩
  rw [(flat_index r.call regions (extent r) h.checks.2.1 idx i active).1,flat,
    Call.output,h.checks.2.1.2.2.2.2.1,Checked.offset_eq r.call.q h.checks.2.1.2.2.2.2.2.1]

#axiomsClean logical_coverage
end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly

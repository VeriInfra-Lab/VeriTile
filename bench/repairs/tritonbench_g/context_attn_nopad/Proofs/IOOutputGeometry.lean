import bench.repairs.tritonbench_g.context_attn_nopad.Proofs.IOData
import bench.tritonbench_g.context_attn_nopad.Proofs.HostCoverage

namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ContextAttnNopad Host Program
noncomputable section
set_option maxHeartbeats 2000000

/-- Every active logical coordinate has a writer in the actual finite grid,
even when maxInputLen does not cover the entire metadata sequence. -/
theorem output_covered (c : Call) (r : Regions) (j : Coordinate c) (on : active c j) :
    FPReference.Writes c r (FPHost.jobs c) r.out (address c c.o j) := by
  rcases j with ⟨b,h,t,d,⟨⟩⟩
  have blockBound : t.val/128<c.blocks := by
    have := t.isLt
    change t.val<c.blocks*128 at this
    omega
  let idx := c.index b.val h.val (t.val/128) b.isLt h.isLt blockBound
  let lane : TileIndex [c.layout.BM,c.layout.D] := (⟨t.val%128,Nat.mod_lt _ (by decide)⟩,d,PUnit.unit)
  refine ⟨idx,(FPHost.jobs_complete c).2 idx,lane,?_,rfl,?_⟩
  · change t.val<c.length b.val at on
    simp only [FPReference.laneActive,idx,index_launch,lane]
    omega
  · simp only [FPReference.laneOffset,address,idx,index_launch,lane]
    congr 2
    omega

theorem outside_written {α : Type} (c : Call) (r : Regions) (o : FPBackend.Ops α)
    (xs : (io c r o).Inputs) (region : RegionName) (off : Nat)
    (outside : (io c r o).Outside xs region off) :
    ¬FPReference.Writes c r (FPHost.jobs c) region off := by
  rintro ⟨idx,_,lane,on,hr,ho⟩
  have jt : (c.launch idx).block*128+lane.1.val<timeExtent c := by
    have := launch_block_lt c idx
    have hl : lane.1.val<128 := lane.1.isLt
    dsimp [timeExtent]
    omega
  let j : Coordinate c := (⟨(c.launch idx).batch,launch_batch_lt c idx⟩,
    ⟨(c.launch idx).head,launch_head_lt c idx⟩,⟨_,jt⟩,lane.2.1,PUnit.unit)
  have h := outside.1 0 j on
  exact h.elim (fun h => h hr) (fun h => h ho)

end
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly.IOData

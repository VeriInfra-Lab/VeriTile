import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.NativeRange
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.Admission

namespace VeriTile.Bench.Repairs.ContextAttnFwd.Native
open VeriTile.Triton VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section
set_option maxHeartbeats 4000000

abbrev batchAt (c : Host.Call) (idx : GridIndex c.grid) := Formula.batch c.layout (c.launchAt idx)
abbrev seqAt (c : Host.Call) (idx : GridIndex c.grid) := c.length (batchAt c idx)+c.prompt (batchAt c idx)
abbrev endAt (c : Host.Call) (idx : GridIndex c.grid) := endIndex (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx))
abbrev stopAt (c : Host.Call) (idx : GridIndex c.grid) := stopIndex (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx))
abbrev countAt (c : Host.Call) (idx : GridIndex c.grid) := blocks (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx))

/-- Descriptor and metadata integer widths checked by the actual wrapper. -/
structure ABI (c : Host.Call) : Prop where
  max_input : c.maxInput<2^31
  fused_grid : c.batches*c.layout.heads≤65535
  start : ∀ b,b<c.batches → c.start b<2^31
  sequence : ∀ b,b<c.batches → c.length b+c.prompt b<2^31

theorem tile (c : Host.Call) : Tile c.layout.BM := by
  unfold Host.Call.layout VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.layout
  simp only [VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.block]
  split <;> simp_all [Tile]

theorem block_bound (c : Host.Call) (idx : GridIndex c.grid) : (c.launchAt idx).block<c.blocks :=
  (idx ⟨0,by simp [Host.Call.grid,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.grid,Grid.rank]⟩).isLt

theorem block_start_bound (c : Host.Call) (idx : GridIndex c.grid) :
    (c.launchAt idx).block*c.layout.BM<c.maxInput :=
  (Bounds.lt_ceilBlocks _ _ _ (Host.bm_positive c)).mp (block_bound c idx)

theorem block_abi (c : Host.Call) (abi : ABI c) (idx : GridIndex c.grid) : (c.launchAt idx).block<2^31 := by
  have h := block_start_bound c idx
  have hm := tile c
  rcases hm with hm|hm <;> rw [hm] at h <;> have := abi.max_input <;> omega

theorem batchHead_abi (c : Host.Call) (abi : ABI c) (idx : GridIndex c.grid) :
    (c.launchAt idx).batchHead<2^31 := by
  have h := (idx ⟨1,by simp [Host.Call.grid,VeriTile.Bench.TritonBenchG.ContextAttnFwd.Host.grid,Grid.rank]⟩).isLt
  change (c.launchAt idx).batchHead<c.batches*c.layout.heads at h
  have := abi.fused_grid
  omega

theorem source_limit (c : Host.Call) (s : BlockState) (hm : Geometry.Metadata c s)
    (idx : GridIndex c.grid) : Formula.limit c.layout (c.launchAt idx) (Program.inputs s c.regions)=(endAt c idx : Int) := by
  have hb := Host.batch_bound c idx
  unfold Formula.limit
  rw [hm.2.1 _ hb,hm.2.2 _ hb]
  simp only [endAt,endIndex,Nat.cast_min,Nat.cast_add,Nat.cast_mul]
  congr 1
  ring

theorem source_count (c : Host.Call) (s : BlockState) (hm : Geometry.Metadata c s)
    (idx : GridIndex c.grid) : Formula.count c.layout (c.launchAt idx) (Program.inputs s c.regions)=countAt c idx := by
  have hb := Host.batch_bound c idx
  rw [Formula.count,Formula.sequence,hm.2.1 _ hb,hm.2.2 _ hb,source_limit c s hm idx]
  simp only [Nat.cast_add,add_sub_cancel_right,←Nat.cast_mul,Int.ofNat_lt,Int.toNat_natCast,c.blockN,
    countAt,blocks,ceilTiles,stopIndex,seqAt,Nat.add_sub_cancel_right]
  split <;> simp_all

theorem native_loop (c : Host.Call) (abi : ABI c) (idx : GridIndex c.grid)
    {α : Type} [Inhabited α] (M : FP.Structural.Algebra α) (s : FP.Structural.State α) :
    bitLoop M (FPSuccess.parts c.layout c.regions (Host.scale c)).body c.layout.BN
      (stopBits (c.launchAt idx).block c.layout.BM (seqAt c idx) (c.prompt (batchAt c idx)))
      (countAt c idx+1) (word 0) s =
    FP.Structural.rangeInt M "start_n" 0 (stopAt c idx : Int) c.layout.BN
      (FPSuccess.parts c.layout c.regions (Host.scale c)).body s := by
  rw [stop_exact _ _ _ _ (block_abi c abi idx) (tile c) (abi.sequence _ (Host.batch_bound c idx)) (by simp only [batchAt]; omega)]
  have hs : stopAt c idx<2^31 := lt_of_le_of_lt (stop_le _ _ _ _) (abi.sequence _ (Host.batch_bound c idx))
  simpa only [c.blockN,countAt,blocks,stopAt] using native_range_eq M
    (FPSuccess.parts c.layout c.regions (Host.scale c)).body (stopAt c idx) c.layout.BM (tile c) hs s

end
end VeriTile.Bench.Repairs.ContextAttnFwd.Native

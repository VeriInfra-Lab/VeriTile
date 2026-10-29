import bench.tritonbench_g.kcache_copy_triton.Proofs.HostPrepare
namespace VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.Probe
open VeriTile.Triton HostGrid
set_option maxRecDepth 16000
set_option maxHeartbeats 1000000
-- Captured from the original Python wrapper AST with actual CPU Torch reshape.
-- Successful rows compare every stride argument, grid, shape, copy/view choice,
-- head dimension and num_warps. Error rows compare the modeled error stage.
def observe (h : Inputs) : Except Error (List Nat × List Nat × Bool × List Nat × List Nat) :=
  (prepare h).map fun p =>
    (p.reshaped.tensor.shape,p.reshaped.tensor.strides,p.reshaped.copied,
      [p.grid.tokens,p.grid.heads,p.grid.splits],
      [p.parameters.strideKt,p.parameters.strideKh,p.parameters.strideKd,
       p.parameters.strideKcb,p.parameters.strideKch,p.parameters.strideKcsplit,
       p.parameters.strideKcs,p.suppliedLastStride,p.parameters.strideBts,p.parameters.strideBtb,
       p.parameters.blockSize,p.parameters.nTokens,p.headDim,p.parameters.width,p.numWarps])

theorem probe_3d_old_n1 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,96,32,0,8,1,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_4d_old_n1 : observe
  { key := ⟨⟨"key"⟩,[2,1,3,8],[24,24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,96,32,0,8,1,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_3d_old_n2 : observe
  { key := ⟨⟨"key"⟩,[4,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 2, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([4,3,8],[24,8,1],Bool.false,[4,3,1],[24,8,1,96,32,0,8,1,4,1,4,2,8,8,4]) := by decide +kernel

theorem probe_4d_old_n2 : observe
  { key := ⟨⟨"key"⟩,[2,2,3,8],[48,24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 2, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([4,3,8],[24,8,1],Bool.false,[4,3,1],[24,8,1,96,32,0,8,1,4,1,4,2,8,8,4]) := by decide +kernel

theorem probe_4d_noncontiguous_copy : observe
  { key := ⟨⟨"key"⟩,[2,2,3,8],[24,48,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 2, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([4,3,8],[24,8,1],Bool.true,[4,3,1],[24,8,1,96,32,0,8,1,4,1,4,2,8,8,4]) := by decide +kernel

theorem probe_4d_rowgap_view : observe
  { key := ⟨⟨"key"⟩,[2,2,3,8],[80,40,10,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 2, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([4,3,8],[40,10,1],Bool.false,[4,3,1],[40,10,1,96,32,0,8,1,4,1,4,2,8,8,4]) := by decide +kernel

theorem probe_4d_leading_singleton_view : observe
  { key := ⟨⟨"key"⟩,[1,2,3,8],[100,24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,96,32,0,8,1,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_4d_tail_singletons : observe
  { key := ⟨⟨"key"⟩,[2,1,1,1],[17,13,7,3]⟩, cache := ⟨⟨"cache"⟩,[4,1,4,1],[4,4,1,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,1,1],[17,3,3],Bool.false,[2,1,1],[17,3,3,4,4,0,1,1,4,1,4,1,1,1,4]) := by decide +kernel

theorem probe_4d_zero_prefix : observe
  { key := ⟨⟨"key"⟩,[0,2,3,8],[48,24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[0],[1]⟩, tables := ⟨⟨"tables"⟩,[0,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([0,3,8],[24,8,1],Bool.false,[0,3,1],[24,8,1,96,32,0,8,1,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_4d_zero_head_ambiguous : observe
  { key := ⟨⟨"key"⟩,[2,1,0,8],[8,8,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,0,4,8],[32,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .reshapeAmbiguous := by decide +kernel

theorem probe_3d_zero_width : observe
  { key := ⟨⟨"key"⟩,[2,3,0],[3,1,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,0],[12,4,1,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .zeroWidth := by decide +kernel

theorem probe_new_zero_width : observe
  { key := ⟨⟨"key"⟩,[2,3,0],[3,1,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,0,4,0],[12,4,4,1,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .error .zeroWidth := by decide +kernel

theorem probe_zero_tokens_n0 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 0, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[0,3,1],[24,8,1,96,32,0,8,1,4,1,4,0,8,8,4]) := by decide +kernel

theorem probe_n3_indivisible : observe
  { key := ⟨⟨"key"⟩,[4,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 3, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .tokenBatch := by decide +kernel

theorem probe_batch_mismatch : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[3],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .incompatibleBatch := by decide +kernel

theorem probe_3d_new_n1 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,2,4,4],[96,32,16,4,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,2],[24,8,1,96,32,16,4,1,4,1,4,1,8,4,4]) := by decide +kernel

theorem probe_3d_new_n2 : observe
  { key := ⟨⟨"key"⟩,[4,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,2,4,4],[96,32,16,4,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 2, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([4,3,8],[24,8,1],Bool.false,[4,3,2],[24,8,1,96,32,16,4,1,4,1,4,2,8,4,4]) := by decide +kernel

theorem probe_4d_new_n2 : observe
  { key := ⟨⟨"key"⟩,[2,2,3,8],[48,24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,2,4,4],[96,32,16,4,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 2, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([4,3,8],[24,8,1],Bool.false,[4,3,2],[24,8,1,96,32,16,4,1,4,1,4,2,8,4,4]) := by decide +kernel

theorem probe_new_wrong_rank : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .error .incompatibleNewCache := by decide +kernel

theorem probe_new_wrong_heads : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,2,2,4,4],[64,32,16,4,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .error .incompatibleNewCache := by decide +kernel

theorem probe_new_wrong_split_product : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,3,4,4],[144,48,16,4,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .error .incompatibleNewCache := by decide +kernel

theorem probe_legacy_accepts_rank5 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,2,4,4],[96,32,16,4,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,96,32,0,16,4,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_legacy_accepts_wrong_heads : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,1,4,8],[32,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,32,32,0,8,1,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_noncontiguous_lengths : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[2]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,96,32,0,8,1,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_extra_length_dimension : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2,2],[2,1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,96,32,0,8,1,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_extra_table_dimension : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4,2],[8,2,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,96,32,0,8,1,8,2,4,1,8,8,4]) := by decide +kernel

theorem probe_warps8_nonpower129 : observe
  { key := ⟨⟨"key"⟩,[2,3,129],[387,129,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,129],[1548,516,129,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,129],[387,129,1],Bool.false,[2,3,1],[387,129,1,1548,516,0,129,1,4,1,4,1,129,129,8]) := by decide +kernel

theorem probe_warps8_power256 : observe
  { key := ⟨⟨"key"⟩,[2,3,256],[768,256,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,256],[3072,1024,256,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,256],[768,256,1],Bool.false,[2,3,1],[768,256,1,3072,1024,0,256,1,4,1,4,1,256,256,8]) := by decide +kernel

theorem probe_old_last_stride2 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[192,64,16,2]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,1],[24,8,1,192,64,0,16,2,4,1,4,1,8,8,4]) := by decide +kernel

theorem probe_new_last_stride2 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,2,4,4],[192,64,32,8,2]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .ok ([2,3,8],[24,8,1],Bool.false,[2,3,2],[24,8,1,192,64,32,8,2,4,1,4,1,8,4,4]) := by decide +kernel

theorem probe_key_rank2 : observe
  { key := ⟨⟨"key"⟩,[2,8],[8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .keyRank := by decide +kernel

theorem probe_cache_rank3 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,8],[24,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.true, reshapeRegion := ⟨"reshape"⟩ } =
  .error .missingCacheStride := by decide +kernel

theorem probe_lengths_rank0 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[],[]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .missingBatchDimension := by decide +kernel

theorem probe_tables_rank1 : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2],[1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp32,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .missingTableStride := by decide +kernel

theorem probe_dtype_mismatch : observe
  { key := ⟨⟨"key"⟩,[2,3,8],[24,8,1]⟩, cache := ⟨⟨"cache"⟩,[4,3,4,8],[96,32,8,1]⟩, lengths := ⟨⟨"lengths"⟩,[2],[1]⟩, tables := ⟨⟨"tables"⟩,[2,4],[4,1]⟩,
    keyProfile := .floating .fp32, cacheProfile := .floating .fp16,
    n := 1, newLayout := Bool.false, reshapeRegion := ⟨"reshape"⟩ } =
  .error .dtypeMismatch := by decide +kernel

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KcacheCopyTriton.Host.Probe

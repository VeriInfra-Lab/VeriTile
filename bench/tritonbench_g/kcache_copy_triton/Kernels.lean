import VeriTile.Triton.DSL

/-! Canonical repaired multi-token source. FP32 data is copied without floating
arithmetic. INT32 metadata and program coordinates are widened before signed
address arithmetic. The candidate commutes the two scalar source offsets. -/
namespace VeriTile.Bench.Repairs.KcacheCopyTriton.Source
open Lean VeriTile.Triton

structure Parameters where
  (K KCache : RegionName)
  (blockTables lengths : Region .int)
  (n strideKt strideKh strideKd strideKcb strideKch strideKcsplit
    strideKcs strideBts strideBtb blockSize width : Nat)

def kernel (commuted : Bool) : Parameters → ComputeKernel :=
  match commuted with
  | Bool.false => fun (p : Parameters) =>
    let K := p.K
    let KCache := p.KCache
    let BLOCK_TABLES := p.blockTables
    let seq_lengths := p.lengths
    triton {
      tl.ports inputs=[K, BLOCK_TABLES.cast, seq_lengths.cast], outputs=[KCache]
      tl.region K = tl.float32, KCache = tl.float32,
        BLOCK_TABLES = tl.int32, seq_lengths = tl.int32
      cur_token_idx = tl.program_id(0).to(tl.int64)
      cur_seq_idx = cur_token_idx // $(p.n)
      cur_token_shift = cur_token_idx - $(p.n) * (cur_seq_idx + $(1))
      cur_kv_head_idx = tl.program_id(1).to(tl.int64)
      split_x_idx = tl.program_id(2).to(tl.int64)
      past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx).to(tl.int64) + cur_token_shift
      last_bt_block_idx = past_kv_seq_len // $(p.blockSize)
      block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(p.strideBts)
      block_id = tl.load(block_table_ptr + last_bt_block_idx * $(p.strideBtb)).to(tl.int64)
      offset_last_block = past_kv_seq_len % $(p.blockSize)
      offsets_dmodel = split_x_idx * $(p.width) + tl.arange(0, $(p.width)).to(tl.int64)
      offsets_k = cur_token_idx * $(p.strideKt) +
        cur_kv_head_idx * $(p.strideKh) + offsets_dmodel * $(p.strideKd)
      k = tl.load(K + offsets_k)
      offsets_kcache = block_id * $(p.strideKcb) +
        cur_kv_head_idx * $(p.strideKch) + split_x_idx * $(p.strideKcsplit) +
        offset_last_block * $(p.strideKcs) + tl.arange(0, $(p.width)).to(tl.int64)
      tl.store(KCache + offsets_kcache, k)
    }
  | Bool.true => fun (p : Parameters) =>
    let K := p.K
    let KCache := p.KCache
    let BLOCK_TABLES := p.blockTables
    let seq_lengths := p.lengths
    triton {
      tl.ports inputs=[K, BLOCK_TABLES.cast, seq_lengths.cast], outputs=[KCache]
      tl.region K = tl.float32, KCache = tl.float32,
        BLOCK_TABLES = tl.int32, seq_lengths = tl.int32
      cur_token_idx = tl.program_id(0).to(tl.int64)
      cur_seq_idx = cur_token_idx // $(p.n)
      cur_token_shift = cur_token_idx - $(p.n) * (cur_seq_idx + $(1))
      cur_kv_head_idx = tl.program_id(1).to(tl.int64)
      split_x_idx = tl.program_id(2).to(tl.int64)
      past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx).to(tl.int64) + cur_token_shift
      last_bt_block_idx = past_kv_seq_len // $(p.blockSize)
      block_table_ptr = BLOCK_TABLES + cur_seq_idx * $(p.strideBts)
      block_id = tl.load(block_table_ptr + last_bt_block_idx * $(p.strideBtb)).to(tl.int64)
      offset_last_block = past_kv_seq_len % $(p.blockSize)
      offsets_dmodel = split_x_idx * $(p.width) + tl.arange(0, $(p.width)).to(tl.int64)
      offsets_k = cur_kv_head_idx * $(p.strideKh) +
        cur_token_idx * $(p.strideKt) + offsets_dmodel * $(p.strideKd)
      k = tl.load(K + offsets_k)
      offsets_kcache = block_id * $(p.strideKcb) +
        cur_kv_head_idx * $(p.strideKch) + split_x_idx * $(p.strideKcsplit) +
        offset_last_block * $(p.strideKcs) + tl.arange(0, $(p.width)).to(tl.int64)
      tl.store(KCache + offsets_kcache, k)
    }

def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.KcacheCopyTriton.Source

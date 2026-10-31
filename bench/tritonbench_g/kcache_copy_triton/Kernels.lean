import VeriTile.Triton.DSL.CallSignature
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

/-- Invocation metadata for the actual source, including both unused arguments. -/
def sourceSignature : CallSignature where
  pointers := [("K",.fp32),("KCache",.fp32),("BLOCK_TABLES",.i32),("seq_lengths",.i32)]
  scalars := [("stride_kt",.integer), ("stride_kh",.integer), ("stride_kd",.integer), ("stride_kcb",.integer), ("stride_kch",.integer), ("stride_kcsplit_x",.integer), ("stride_kcs",.integer), ("stride_kcx",.integer), ("stride_bts",.integer), ("stride_btb",.integer), ("block_size",.integer), ("n_tokens",.integer)]
  constexpr := ["HEAD_DIM","KCACHE_X"]
  readPorts := ["K","BLOCK_TABLES","seq_lengths"]
  writePorts := ["KCache"]

/-- Complete actual Python body. The optional arguments retain the old layout
API; both are unused by the body, and the complete program is independent of them. -/
def quotedOriginal (p : Parameters) (stride_kcx : Nat := 1) (HEAD_DIM : Nat := p.width) : ComputeKernel :=
  let K := p.K
  let KCache := p.KCache
  let BLOCK_TABLES := p.blockTables
  let seq_lengths := p.lengths
  let KCACHE_X := p.width
  with_triton_call sourceSignature call_args [stride_kt := p.strideKt, stride_kh := p.strideKh, stride_kd := p.strideKd, stride_kcb := p.strideKcb, stride_kch := p.strideKch, stride_kcsplit_x := p.strideKcsplit, stride_kcs := p.strideKcs, stride_kcx := stride_kcx, stride_bts := p.strideBts, stride_btb := p.strideBtb, block_size := p.blockSize, n_tokens := p.n] in
  triton {
    # `n_tokens` is used to specify the number of tokens to copy for each sequence
    # When n_tokens > 1, tokens from different sequences are packed into the first dimension of the grid,
    #   `seq_lengths` must be the lengths of sequences counting the number of tokens to copy
    #   E.g. if n_tokens = 5, seq_lengths = [12, 15], then the already-copied position ids are [0-6, 0-9]
    #   for the two sequences, respectively. And the position ids to be copied are [7-11, 9-14].
    # When n_tokens = 1, consider token idx as the sequence idx, since it's only used during regular decoding stage
    cur_token_idx = tl.program_id(0).to(tl.int64)
    cur_seq_idx = cur_token_idx // n_tokens
    # `cur_token_shift` is only valid and functional when `n_tokens` > 1
    cur_token_shift = cur_token_idx - (n_tokens * (cur_seq_idx + 1))
    cur_kv_head_idx = tl.program_id(1).to(tl.int64)
    split_x_idx = tl.program_id(2).to(tl.int64)

    past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx).to(tl.int64) + cur_token_shift
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb).to(tl.int64)
    offset_last_block = past_kv_seq_len % block_size
    offsets_dmodel = split_x_idx * KCACHE_X + tl.arange(0, KCACHE_X).to(tl.int64)
    offsets_k = cur_token_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel * stride_kd
    k = tl.load(K + offsets_k)
    offsets_kcache = (
        block_id * stride_kcb
        + cur_kv_head_idx * stride_kch
        + split_x_idx * stride_kcsplit_x
        + offset_last_block * stride_kcs
        + tl.arange(0, KCACHE_X).to(tl.int64)
    )
    tl.store(KCache + offsets_kcache, k)
    return
  }

/-- Complete actual Python body. The optional arguments retain the old layout
API; both are unused by the body, and the complete program is independent of them. -/
def quotedCommuted (p : Parameters) (stride_kcx : Nat := 1) (HEAD_DIM : Nat := p.width) : ComputeKernel :=
  let K := p.K
  let KCache := p.KCache
  let BLOCK_TABLES := p.blockTables
  let seq_lengths := p.lengths
  let KCACHE_X := p.width
  with_triton_call sourceSignature call_args [stride_kt := p.strideKt, stride_kh := p.strideKh, stride_kd := p.strideKd, stride_kcb := p.strideKcb, stride_kch := p.strideKch, stride_kcsplit_x := p.strideKcsplit, stride_kcs := p.strideKcs, stride_kcx := stride_kcx, stride_bts := p.strideBts, stride_btb := p.strideBtb, block_size := p.blockSize, n_tokens := p.n] in
  triton {
    # `n_tokens` is used to specify the number of tokens to copy for each sequence
    # When n_tokens > 1, tokens from different sequences are packed into the first dimension of the grid,
    #   `seq_lengths` must be the lengths of sequences counting the number of tokens to copy
    #   E.g. if n_tokens = 5, seq_lengths = [12, 15], then the already-copied position ids are [0-6, 0-9]
    #   for the two sequences, respectively. And the position ids to be copied are [7-11, 9-14].
    # When n_tokens = 1, consider token idx as the sequence idx, since it's only used during regular decoding stage
    cur_token_idx = tl.program_id(0).to(tl.int64)
    cur_seq_idx = cur_token_idx // n_tokens
    # `cur_token_shift` is only valid and functional when `n_tokens` > 1
    cur_token_shift = cur_token_idx - (n_tokens * (cur_seq_idx + 1))
    cur_kv_head_idx = tl.program_id(1).to(tl.int64)
    split_x_idx = tl.program_id(2).to(tl.int64)

    past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx).to(tl.int64) + cur_token_shift
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb).to(tl.int64)
    offset_last_block = past_kv_seq_len % block_size
    offsets_dmodel = split_x_idx * KCACHE_X + tl.arange(0, KCACHE_X).to(tl.int64)
    offsets_k = cur_kv_head_idx * stride_kh + cur_token_idx * stride_kt + offsets_dmodel * stride_kd
    k = tl.load(K + offsets_k)
    offsets_kcache = (
        block_id * stride_kcb
        + cur_kv_head_idx * stride_kch
        + split_x_idx * stride_kcsplit_x
        + offset_last_block * stride_kcs
        + tl.arange(0, KCACHE_X).to(tl.int64)
    )
    tl.store(KCache + offsets_kcache, k)
    return
  }

def kernel (commuted : Bool) (p : Parameters) : ComputeKernel :=
  if commuted then quotedCommuted p else quotedOriginal p

def originalKernel := kernel Bool.false
def commutedKernel := kernel Bool.true

end VeriTile.Bench.Repairs.KcacheCopyTriton.Source

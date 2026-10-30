import VeriTile.Triton.DSL.CallSignature
import VeriTile.Triton.DSL

/-! Actual selected Python function bodies; storage and argument roles live outside.
`address_shared.py` is newly authored only for the existing Lean Q-address CSE. -/
namespace VeriTile.Bench.Repairs.RotaryEmbNopad.Kernels
open VeriTile.Triton

structure UncachedLayout where
  (q k cos sin : RegionName)
  (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
    cos_token_stride cos_stride sin_token_stride sin_stride q_total_tokens
    Q_HEAD_NUM KV_GROUP_NUM HEAD_DIM BLOCK_PAIRS BLOCK_TOKENS : Nat)

structure CachedLayout where
  (q k cos sin kv_cache BLOCK_TABLES context_lengths : RegionName)
  (q_token_stride q_head_stride k_token_stride k_head_stride head_dim_stride
    cos_token_stride cos_stride sin_token_stride sin_stride cacheb_stride
    cacheh_stride cachebs_stride cached_stride bts_stride btb_stride
    block_size q_total_tokens Q_HEAD_NUM K_HEAD_NUM HEAD_DIM
    BLOCK_PAIRS : Nat)

/-- Exact selected source ABI; runtime strides/counts and constexpr dimensions. -/
def uncachedSignature : CallSignature where
  pointers := [("q",.fp32), ("k",.fp32), ("cos",.fp32), ("sin",.fp32)]
  scalars := [("q_token_stride",.integer), ("q_head_stride",.integer), ("k_token_stride",.integer), ("k_head_stride",.integer), ("head_dim_stride",.integer), ("cos_token_stride",.integer), ("cos_stride",.integer), ("sin_token_stride",.integer), ("sin_stride",.integer), ("q_total_tokens",.integer)]
  constexpr := ["Q_HEAD_NUM","KV_GROUP_NUM","HEAD_DIM","BLOCK_PAIRS","BLOCK_TOKENS"]
  readPorts := ["q","k","cos","sin"]
  writePorts := ["q","k"]

/-- Complete selected original body. -/
def uncached (p : UncachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  let Q_HEAD_NUM := p.Q_HEAD_NUM
  let KV_GROUP_NUM := p.KV_GROUP_NUM
  let HEAD_DIM := p.HEAD_DIM
  let BLOCK_PAIRS := p.BLOCK_PAIRS
  let BLOCK_TOKENS := p.BLOCK_TOKENS
  let q_token_stride : Int := p.q_token_stride
  let q_head_stride : Int := p.q_head_stride
  let k_token_stride : Int := p.k_token_stride
  let k_head_stride : Int := p.k_head_stride
  let head_dim_stride : Int := p.head_dim_stride
  let cos_token_stride : Int := p.cos_token_stride
  let cos_stride : Int := p.cos_stride
  let sin_token_stride : Int := p.sin_token_stride
  let sin_stride : Int := p.sin_stride
  let q_total_tokens : Int := p.q_total_tokens
  with_triton_call uncachedSignature call_args [q_token_stride := q_token_stride, q_head_stride := q_head_stride, k_token_stride := k_token_stride, k_head_stride := k_head_stride, head_dim_stride := head_dim_stride, cos_token_stride := cos_token_stride, cos_stride := cos_stride, sin_token_stride := sin_token_stride, sin_stride := sin_stride, q_total_tokens := q_total_tokens] in
  triton {
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_block_idx = tl.program_id(1).to(tl.int64)

    tokens_range = cur_token_block_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS).to(tl.int64)
    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2

    off_cos_sin = tokens_range[:, None] * cos_token_stride + dim_range0[None, :] * cos_stride
    loaded_cos = tl.load(cos + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens) & (dim_range0[None, :] < HEAD_DIM // 2), other=0.0)
    off_sin = tokens_range[:, None] * sin_token_stride + dim_range0[None, :] * sin_stride
    loaded_sin = tl.load(sin + off_sin, mask=(tokens_range[:, None] < q_total_tokens) & (dim_range0[None, :] < HEAD_DIM // 2), other=0.0)

    off_q0 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_q1 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )
    loaded_q0 = tl.load(
        q + off_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
        other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
        other=0.0,
    )
    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

    tl.store(
        q + off_q0,
        out_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
    )
    tl.store(
        q + off_q1,
        out_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
    )

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k0 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range0[None, None, :] * head_dim_stride
        )
        off_k1 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range1[None, None, :] * head_dim_stride
        )
        loaded_k0 = tl.load(
            k + off_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
            other=0.0,
        )
        loaded_k1 = tl.load(
            k + off_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
            other=0.0,
        )
        out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
        out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
        tl.store(
            k + off_k0,
            out_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
        )
        tl.store(
            k + off_k1,
            out_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
        )
  }

/-- Complete pre-existing index candidate with explicitly new Python counterpart body. -/
def uncachedShared (p : UncachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  let Q_HEAD_NUM := p.Q_HEAD_NUM
  let KV_GROUP_NUM := p.KV_GROUP_NUM
  let HEAD_DIM := p.HEAD_DIM
  let BLOCK_PAIRS := p.BLOCK_PAIRS
  let BLOCK_TOKENS := p.BLOCK_TOKENS
  let q_token_stride : Int := p.q_token_stride
  let q_head_stride : Int := p.q_head_stride
  let k_token_stride : Int := p.k_token_stride
  let k_head_stride : Int := p.k_head_stride
  let head_dim_stride : Int := p.head_dim_stride
  let cos_token_stride : Int := p.cos_token_stride
  let cos_stride : Int := p.cos_stride
  let sin_token_stride : Int := p.sin_token_stride
  let sin_stride : Int := p.sin_stride
  let q_total_tokens : Int := p.q_total_tokens
  with_triton_call uncachedSignature call_args [q_token_stride := q_token_stride, q_head_stride := q_head_stride, k_token_stride := k_token_stride, k_head_stride := k_head_stride, head_dim_stride := head_dim_stride, cos_token_stride := cos_token_stride, cos_stride := cos_stride, sin_token_stride := sin_token_stride, sin_stride := sin_stride, q_total_tokens := q_total_tokens] in
  triton {
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_block_idx = tl.program_id(1).to(tl.int64)

    tokens_range = cur_token_block_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS).to(tl.int64)
    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2

    off_cos_sin = tokens_range[:, None] * cos_token_stride + dim_range0[None, :] * cos_stride
    loaded_cos = tl.load(cos + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens) & (dim_range0[None, :] < HEAD_DIM // 2), other=0.0)
    off_sin = tokens_range[:, None] * sin_token_stride + dim_range0[None, :] * sin_stride
    loaded_sin = tl.load(sin + off_sin, mask=(tokens_range[:, None] < q_total_tokens) & (dim_range0[None, :] < HEAD_DIM // 2), other=0.0)

    off_q0 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_q1 = off_q0 + (HEAD_DIM // 2) * head_dim_stride
    loaded_q0 = tl.load(
        q + off_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
        other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
        other=0.0,
    )
    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

    tl.store(
        q + off_q0,
        out_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
    )
    tl.store(
        q + off_q1,
        out_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
    )

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k0 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range0[None, None, :] * head_dim_stride
        )
        off_k1 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range1[None, None, :] * head_dim_stride
        )
        loaded_k0 = tl.load(
            k + off_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
            other=0.0,
        )
        loaded_k1 = tl.load(
            k + off_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
            other=0.0,
        )
        out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
        out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
        tl.store(
            k + off_k0,
            out_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
        )
        tl.store(
            k + off_k1,
            out_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
        )
  }
/-- Exact selected source ABI; runtime strides/counts and constexpr dimensions. -/
def cachedSignature : CallSignature where
  pointers := [("q",.fp32), ("k",.fp32), ("cos",.fp32), ("sin",.fp32), ("kv_cache",.fp32), ("BLOCK_TABLES",.i64), ("context_lengths",.i64)]
  scalars := [("q_token_stride",.integer), ("q_head_stride",.integer), ("k_token_stride",.integer), ("k_head_stride",.integer), ("head_dim_stride",.integer), ("cos_token_stride",.integer), ("cos_stride",.integer), ("sin_token_stride",.integer), ("sin_stride",.integer), ("cacheb_stride",.integer), ("cacheh_stride",.integer), ("cachebs_stride",.integer), ("cached_stride",.integer), ("bts_stride",.integer), ("btb_stride",.integer), ("block_size",.integer), ("q_total_tokens",.integer)]
  constexpr := ["Q_HEAD_NUM","K_HEAD_NUM","HEAD_DIM","BLOCK_PAIRS"]
  readPorts := ["q","k","cos","sin","kv_cache","BLOCK_TABLES","context_lengths"]
  writePorts := ["q","kv_cache"]

/-- Complete selected original body. -/
def cached (p : CachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  let kv_cache := p.kv_cache
  let BLOCK_TABLES := p.BLOCK_TABLES
  let context_lengths := p.context_lengths
  let Q_HEAD_NUM := p.Q_HEAD_NUM
  let K_HEAD_NUM := p.K_HEAD_NUM
  let HEAD_DIM := p.HEAD_DIM
  let BLOCK_PAIRS := p.BLOCK_PAIRS
  let q_token_stride : Int := p.q_token_stride
  let q_head_stride : Int := p.q_head_stride
  let k_token_stride : Int := p.k_token_stride
  let k_head_stride : Int := p.k_head_stride
  let head_dim_stride : Int := p.head_dim_stride
  let cos_token_stride : Int := p.cos_token_stride
  let cos_stride : Int := p.cos_stride
  let sin_token_stride : Int := p.sin_token_stride
  let sin_stride : Int := p.sin_stride
  let cacheb_stride : Int := p.cacheb_stride
  let cacheh_stride : Int := p.cacheh_stride
  let cachebs_stride : Int := p.cachebs_stride
  let cached_stride : Int := p.cached_stride
  let bts_stride : Int := p.bts_stride
  let btb_stride : Int := p.btb_stride
  let block_size : Int := p.block_size
  let q_total_tokens : Int := p.q_total_tokens
  with_triton_call cachedSignature call_args [q_token_stride := q_token_stride, q_head_stride := q_head_stride, k_token_stride := k_token_stride, k_head_stride := k_head_stride, head_dim_stride := head_dim_stride, cos_token_stride := cos_token_stride, cos_stride := cos_stride, sin_token_stride := sin_token_stride, sin_stride := sin_stride, cacheb_stride := cacheb_stride, cacheh_stride := cacheh_stride, cachebs_stride := cachebs_stride, cached_stride := cached_stride, bts_stride := bts_stride, btb_stride := btb_stride, block_size := block_size, q_total_tokens := q_total_tokens] in
  triton {
    block_head_index = tl.program_id(0).to(tl.int64)
    if block_head_index >= Q_HEAD_NUM:
        return
    block_token_index = tl.program_id(1).to(tl.int64)

    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2

    off_q0 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range0 * head_dim_stride
    off_q1 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range1 * head_dim_stride
    off_k0 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range0 * head_dim_stride
    off_k1 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range1 * head_dim_stride

    loaded_q0 = tl.load(
        q + off_q0, mask=dim_range0 < HEAD_DIM // 2, other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1, mask=dim_range0 < HEAD_DIM // 2, other=0.0,
    )

    loaded_k0 = tl.load(
        k + off_k0, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2), other=0.0,
    )

    loaded_k1 = tl.load(
        k + off_k1, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2), other=0.0,
    )

    off_cos_sin = block_token_index * cos_token_stride + dim_range0 * cos_stride

    loaded_cos = tl.load(cos + off_cos_sin, mask=(block_token_index < q_total_tokens) & (dim_range0 < HEAD_DIM // 2), other=0.0)
    off_sin = block_token_index * sin_token_stride + dim_range0 * sin_stride
    loaded_sin = tl.load(sin + off_sin, mask=(block_token_index < q_total_tokens) & (dim_range0 < HEAD_DIM // 2), other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos  # total_tokens, head_num, head_dim

    past_kv_seq_len = tl.load(context_lengths + block_token_index) - 1

    last_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + block_token_index * bts_stride
    block_ids = tl.load(block_table_ptr + last_block_idx * btb_stride, mask=(block_token_index < q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride

    kv_range0 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range0 * cached_stride
    )
    kv_range1 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range1 * cached_stride
    )

    tl.store(
        kv_cache + kv_range0,
        out_k0, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2),
    )
    tl.store(
        kv_cache + kv_range1,
        out_k1, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2),
    )

    # concat
    tl.store(
        q + off_q0,
        out_q0, mask=dim_range0 < HEAD_DIM // 2,
    )
    tl.store(
        q + off_q1,
        out_q1, mask=dim_range0 < HEAD_DIM // 2,
    )
  }

/-- Complete pre-existing index candidate with explicitly new Python counterpart body. -/
def cachedShared (p : CachedLayout) : ComputeKernel :=
  let q := p.q
  let k := p.k
  let cos := p.cos
  let sin := p.sin
  let kv_cache := p.kv_cache
  let BLOCK_TABLES := p.BLOCK_TABLES
  let context_lengths := p.context_lengths
  let Q_HEAD_NUM := p.Q_HEAD_NUM
  let K_HEAD_NUM := p.K_HEAD_NUM
  let HEAD_DIM := p.HEAD_DIM
  let BLOCK_PAIRS := p.BLOCK_PAIRS
  let q_token_stride : Int := p.q_token_stride
  let q_head_stride : Int := p.q_head_stride
  let k_token_stride : Int := p.k_token_stride
  let k_head_stride : Int := p.k_head_stride
  let head_dim_stride : Int := p.head_dim_stride
  let cos_token_stride : Int := p.cos_token_stride
  let cos_stride : Int := p.cos_stride
  let sin_token_stride : Int := p.sin_token_stride
  let sin_stride : Int := p.sin_stride
  let cacheb_stride : Int := p.cacheb_stride
  let cacheh_stride : Int := p.cacheh_stride
  let cachebs_stride : Int := p.cachebs_stride
  let cached_stride : Int := p.cached_stride
  let bts_stride : Int := p.bts_stride
  let btb_stride : Int := p.btb_stride
  let block_size : Int := p.block_size
  let q_total_tokens : Int := p.q_total_tokens
  with_triton_call cachedSignature call_args [q_token_stride := q_token_stride, q_head_stride := q_head_stride, k_token_stride := k_token_stride, k_head_stride := k_head_stride, head_dim_stride := head_dim_stride, cos_token_stride := cos_token_stride, cos_stride := cos_stride, sin_token_stride := sin_token_stride, sin_stride := sin_stride, cacheb_stride := cacheb_stride, cacheh_stride := cacheh_stride, cachebs_stride := cachebs_stride, cached_stride := cached_stride, bts_stride := bts_stride, btb_stride := btb_stride, block_size := block_size, q_total_tokens := q_total_tokens] in
  triton {
    block_head_index = tl.program_id(0).to(tl.int64)
    if block_head_index >= Q_HEAD_NUM:
        return
    block_token_index = tl.program_id(1).to(tl.int64)

    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2

    off_q0 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range0 * head_dim_stride
    off_q1 = off_q0 + (HEAD_DIM // 2) * head_dim_stride
    off_k0 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range0 * head_dim_stride
    off_k1 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range1 * head_dim_stride

    loaded_q0 = tl.load(
        q + off_q0, mask=dim_range0 < HEAD_DIM // 2, other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1, mask=dim_range0 < HEAD_DIM // 2, other=0.0,
    )

    loaded_k0 = tl.load(
        k + off_k0, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2), other=0.0,
    )

    loaded_k1 = tl.load(
        k + off_k1, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2), other=0.0,
    )

    off_cos_sin = block_token_index * cos_token_stride + dim_range0 * cos_stride

    loaded_cos = tl.load(cos + off_cos_sin, mask=(block_token_index < q_total_tokens) & (dim_range0 < HEAD_DIM // 2), other=0.0)
    off_sin = block_token_index * sin_token_stride + dim_range0 * sin_stride
    loaded_sin = tl.load(sin + off_sin, mask=(block_token_index < q_total_tokens) & (dim_range0 < HEAD_DIM // 2), other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos  # total_tokens, head_num, head_dim

    past_kv_seq_len = tl.load(context_lengths + block_token_index) - 1

    last_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + block_token_index * bts_stride
    block_ids = tl.load(block_table_ptr + last_block_idx * btb_stride, mask=(block_token_index < q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride

    kv_range0 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range0 * cached_stride
    )
    kv_range1 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range1 * cached_stride
    )

    tl.store(
        kv_cache + kv_range0,
        out_k0, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2),
    )
    tl.store(
        kv_cache + kv_range1,
        out_k1, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2),
    )

    # concat
    tl.store(
        q + off_q0,
        out_q0, mask=dim_range0 < HEAD_DIM // 2,
    )
    tl.store(
        q + off_q1,
        out_q1, mask=dim_range0 < HEAD_DIM // 2,
    )
  }

end VeriTile.Bench.Repairs.RotaryEmbNopad.Kernels

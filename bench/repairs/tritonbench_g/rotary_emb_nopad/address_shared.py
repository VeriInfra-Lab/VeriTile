"""New Python counterparts of the existing Lean Q-address CSE candidates.

The authoritative original remains rotary_emb_nopad_repaired.py unchanged.
Each candidate replaces only off_q1 with off_q0 + (HEAD_DIM // 2) * head_dim_stride.
All numerical expressions, memory masks, control flow and stores remain original.
"""
import triton
import triton.language as tl

@triton.jit
def rotary_embedding_kernel_shared(
    q,
    k,
    cos,
    sin,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    sin_token_stride,
    sin_stride,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,  # token range length
):
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

@triton.jit
def fused_rotary_embedding_kernel_v2_shared(
    q,
    k,
    cos,
    sin,
    kv_cache,
    BLOCK_TABLES,
    context_lengths,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    sin_token_stride,
    sin_stride,
    cacheb_stride,
    cacheh_stride,
    cachebs_stride,
    cached_stride,
    bts_stride,
    btb_stride,
    block_size,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    K_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
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


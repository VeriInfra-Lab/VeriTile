"""Authored from the existing Lean shared-Q-offset candidate.

Only the second Q address reuses off_q0 and the half-head displacement.
All K/V work, floating arithmetic, memory operations and masks are retained.
The selected original JIT and its wrapper remain in the original source file.
"""
import triton
import triton.language as tl

@triton.jit
def decoding_fused_rotary_embedding_shared_q_offset(
    q, k, v, cos, sin, k_cache, v_cache, BLOCK_TABLES, context_lengths,
    x, q_token_stride, q_head_stride, q_dim_stride,
    k_token_stride, k_head_stride, k_dim_stride,
    v_token_stride, v_head_stride, v_dim_stride,
    cos_token_stride, cos_stride, sin_token_stride, sin_stride,
    length_stride, kcb_stride, kch_stride, kcsplit_x_stride, kcs_stride,
    kcd_stride, vcb_stride, vch_stride, vcs_stride, vcd_stride,
    bts_stride, btb_stride, block_size,
    KV_GROUP_NUM: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_idx = tl.program_id(1).to(tl.int64)
    dim_range = tl.arange(0, BLOCK_DIM).to(tl.int64)
    dim_range0 = tl.arange(0, BLOCK_HALF).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2
    pair_mask = dim_range0 < HEAD_DIM // 2

    off_q = cur_token_idx * q_token_stride + cur_head_idx * q_head_stride
    off_q0 = off_q + dim_range0 * q_dim_stride
    off_q1 = off_q0 + (HEAD_DIM // 2) * q_dim_stride
    loaded_q0 = tl.load(q + off_q0, mask=pair_mask, other=0)
    loaded_q1 = tl.load(q + off_q1, mask=pair_mask, other=0)
    off_cos = cur_token_idx * cos_token_stride + dim_range0 * cos_stride
    off_sin = cur_token_idx * sin_token_stride + dim_range0 * sin_stride
    loaded_cos = tl.load(cos + off_cos, mask=pair_mask, other=0)
    loaded_sin = tl.load(sin + off_sin, mask=pair_mask, other=0)
    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0, mask=pair_mask)
    tl.store(q + off_q1, out_q1, mask=pair_mask)

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        cur_k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k = cur_token_idx * k_token_stride + cur_k_head_idx * k_head_stride
        off_k0 = off_k + dim_range0 * k_dim_stride
        off_k1 = off_k + dim_range1 * k_dim_stride
        loaded_k0 = tl.load(k + off_k0, mask=pair_mask, other=0)
        loaded_k1 = tl.load(k + off_k1, mask=pair_mask, other=0)
        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx * length_stride).to(tl.int64) - 1
        last_block_idx = past_kv_seq_len // block_size
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * bts_stride + last_block_idx * btb_stride).to(tl.int64)
        offsets_in_last_block = past_kv_seq_len % block_size
        offsets_cache_base = block_ids * kcb_stride + cur_k_head_idx * kch_stride
        k_range0 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range0 // x) * kcsplit_x_stride
                    + (dim_range0 % x) * kcd_stride)
        k_range1 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range1 // x) * kcsplit_x_stride
                    + (dim_range1 % x) * kcd_stride)
        tl.store(k_cache + k_range0, out_k0, mask=pair_mask)
        tl.store(k_cache + k_range1, out_k1, mask=pair_mask)
        off_v = (cur_token_idx * v_token_stride + cur_k_head_idx * v_head_stride
                 + dim_range * v_dim_stride)
        loaded_v = tl.load(v + off_v, mask=dim_range < HEAD_DIM, other=0)
        v_range = (block_ids * vcb_stride + cur_k_head_idx * vch_stride
                   + offsets_in_last_block * vcs_stride + dim_range * vcd_stride)
        tl.store(v_cache + v_range, loaded_v, mask=dim_range < HEAD_DIM)

"""New Python counterpart of the existing Lean integer index candidate.
Original source remains rotary_emb_repaired.py unchanged. Only the head block
product is commuted; numerical operations, masks and stores retain their order.
"""
import triton
import triton.language as tl

@triton.jit
def _rotary_kernel_index_commuted(
    Q,
    K,
    Cos,
    Sin,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_cosbs,
    stride_cosd,
    stride_sinbs,
    stride_sind,
    max_total_len,
    ROTARY_DIM,
    HEAD_Q,
    HEAD_K,
    BLOCK_HEAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    cur_head_index = tl.program_id(0).to(tl.int64)
    cur_seq_index = tl.program_id(1).to(tl.int64)

    cur_head_range = BLOCK_HEAD * cur_head_index + tl.arange(0, BLOCK_HEAD).to(tl.int64)
    cur_seq_range = cur_seq_index * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ).to(tl.int64)

    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64) * 2
    dim_range1 = tl.arange(0, BLOCK_PAIRS).to(tl.int64) * 2 + 1

    off_q0 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range0[None, None, :] * stride_qd
    )
    off_q1 = (
        cur_seq_range[:, None, None] * stride_qbs
        + cur_head_range[None, :, None] * stride_qh
        + dim_range1[None, None, :] * stride_qd
    )

    off_dimcos_sin0 = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    off_dimcos_sin1 = cur_seq_range[:, None, None] * stride_cosbs + dim_range1[None, None, :] * stride_cosd

    off_sin0 = cur_seq_range[:, None, None] * stride_sinbs + dim_range0[None, None, :] * stride_sind
    off_sin1 = cur_seq_range[:, None, None] * stride_sinbs + dim_range1[None, None, :] * stride_sind

    q0 = tl.load(
        Q + off_q0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )
    q1 = tl.load(
        Q + off_q1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )

    cos0 = tl.load(Cos + off_dimcos_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin0 = tl.load(Sin + off_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    cos1 = tl.load(Cos + off_dimcos_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin1 = tl.load(Sin + off_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    out0 = q0 * cos0 - q1 * sin0
    out1 = q0 * sin1 + q1 * cos1

    tl.store(
        Q + off_q0, out0, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM)
    )
    tl.store(
        Q + off_q1, out1, mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_Q) & (dim_range1[None, None, :] < ROTARY_DIM)
    )

    off_k0 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range0[None, None, :] * stride_kd
    )
    off_k1 = (
        cur_seq_range[:, None, None] * stride_kbs
        + cur_head_range[None, :, None] * stride_kh
        + dim_range1[None, None, :] * stride_kd
    )

    off_dimcos_sin0 = cur_seq_range[:, None, None] * stride_cosbs + dim_range0[None, None, :] * stride_cosd
    off_dimcos_sin1 = cur_seq_range[:, None, None] * stride_cosbs + dim_range1[None, None, :] * stride_cosd

    k0 = tl.load(
        K + off_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )
    k1 = tl.load(
        K + off_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
        other=0.0,
    )

    cos0 = tl.load(Cos + off_dimcos_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin0 = tl.load(Sin + off_sin0, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    cos1 = tl.load(Cos + off_dimcos_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)
    sin1 = tl.load(Sin + off_sin1, mask=(cur_seq_range[:, None, None] < max_total_len) & (dim_range1[None, None, :] < ROTARY_DIM), other=0.0)

    out_k0 = k0 * cos0 - k1 * sin0
    out_k1 = k0 * sin1 + k1 * cos1

    tl.store(
        K + off_k0,
        out_k0,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
    )
    tl.store(
        K + off_k1,
        out_k1,
        mask=(cur_seq_range[:, None, None] < max_total_len) & (cur_head_range[None, :, None] < HEAD_K) & (dim_range1[None, None, :] < ROTARY_DIM),
    )
    return

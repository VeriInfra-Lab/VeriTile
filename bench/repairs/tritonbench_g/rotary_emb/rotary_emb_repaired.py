"""Interleaved in-place rotary embedding for ordinary separate FP32 CUDA allocations.

Q/K are canonical contiguous [tokens, heads, width] tensors, with equal width
and possibly different positive head counts. Cos/Sin are separately allocated
canonical contiguous [tokens, coefficient_width] tensors, with independent row
strides and enough columns for the selected even rotary width. All FP32 payloads
are admitted. The function returns None, preserves tensor identity and untouched
columns, and retains the default compiler fusion policy.
"""
import math
import triton
import triton.language as tl
import torch

@triton.jit
def _rotary_kernel(
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

    cur_head_range = cur_head_index * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD).to(tl.int64)
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

@torch.no_grad()
def rotary_emb_fwd(q, k, cos, sin, partial_rotary_factor=1.0):
    tensors = (q, k, cos, sin)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("q, k, cos and sin must be tensors")
    if not all(x.layout == torch.strided and x.dtype == torch.float32 and x.is_cuda for x in tensors):
        raise TypeError("ordinary strided CUDA FP32 tensors are required")
    if not all(x.device == q.device for x in tensors):
        raise ValueError("all tensors must be on the same device")
    if q.ndim != 3 or k.ndim != 3 or cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("Q/K must have rank 3 and Cos/Sin rank 2")
    total_len, head_num_q, width = q.shape
    head_num_k = k.shape[1]
    if head_num_q <= 0 or head_num_k <= 0 or width <= 0:
        raise ValueError("positive head counts and head width are required")
    if k.shape[0] != total_len or k.shape[2] != width or cos.shape[0] != total_len or sin.shape[0] != total_len:
        raise ValueError("sequence lengths and Q/K head widths must agree")
    if type(partial_rotary_factor) not in (int, float) or not math.isfinite(partial_rotary_factor) or not 0 <= partial_rotary_factor <= 1:
        raise ValueError("partial_rotary_factor must be finite and between zero and one")
    rotary_dim = int(width * partial_rotary_factor)
    if rotary_dim > width or rotary_dim % 2:
        raise ValueError("the selected rotary width must be even and no greater than the head width")
    if cos.shape[1] < rotary_dim or sin.shape[1] < rotary_dim:
        raise ValueError("coefficient tables need every rotated column")
    if any(not x.is_contiguous() or x.storage_offset() != 0 or x.is_neg() or x.is_conj() for x in tensors):
        raise ValueError("canonical contiguous tensors without view offsets or lazy flags are required")
    if any(x.numel() > (2**63 - 1) // 4 for x in tensors):
        raise ValueError("byte offsets must fit the signed 64-bit address contract")
    if total_len == 0 or rotary_dim == 0:
        return None
    expected = ((head_num_q * width, width, 1), (head_num_k * width, width, 1),
                (cos.shape[1], 1), (sin.shape[1], 1))
    if any(tuple(x.stride()) != stride for x, stride in zip(tensors, expected)):
        raise ValueError("canonical contiguous strides are required")
    spans = []
    for x in tensors:
        storage = x.untyped_storage()
        start, size = storage.data_ptr(), storage.nbytes()
        if (x.data_ptr() != start or start % 4 or
                4 * x.numel() > size or start + size > 2**64):
            raise ValueError("aligned contained FP32 storage is required")
        spans.append((start, start + size))
    if any(a < d and c < b for i, (a, b) in enumerate(spans)
           for c, d in spans[i + 1:]):
        raise ValueError("Q, K, Cos and Sin backing allocations must be separate")
    block_pairs = triton.next_power_of_2(rotary_dim // 2)
    grid = (triton.cdiv(max(head_num_q, head_num_k), 4), triton.cdiv(total_len, 16))
    padded_tokens, padded_heads = grid[1] * 16, grid[0] * 4
    last_dim = 2 * block_pairs - 1
    last_offsets = (
        (padded_tokens - 1) * head_num_q * width + (padded_heads - 1) * width + last_dim,
        (padded_tokens - 1) * head_num_k * width + (padded_heads - 1) * width + last_dim,
        (padded_tokens - 1) * cos.shape[1] + last_dim,
        (padded_tokens - 1) * sin.shape[1] + last_dim,
    )
    if any(v >= 2**63 for v in (total_len, head_num_q, head_num_k, width,
                                cos.shape[1], sin.shape[1], padded_tokens,
                                padded_heads, 2 * block_pairs, head_num_q * width,
                                head_num_k * width)):
        raise ValueError("all source integer extents must fit signed int64")
    if any(4 * last + 4 >= 2**63 or x.data_ptr() + 4 * last + 4 > 2**64
           for x, last in zip(tensors, last_offsets)):
        raise ValueError("padded source offsets and formed pointers must fit")
    with torch.cuda.device(q.device):
        _rotary_kernel[grid](
            q, k, cos, sin,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            total_len, rotary_dim, head_num_q, head_num_k,
            BLOCK_HEAD=4, BLOCK_SEQ=16, BLOCK_PAIRS=block_pairs,
            num_warps=8 if rotary_dim >= 128 else 4, num_stages=1,
        )
    return None

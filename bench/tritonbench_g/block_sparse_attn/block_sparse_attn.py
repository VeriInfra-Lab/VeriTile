
import math
import torch
import triton
import triton.language as tl

def _storage(t, name):
    """Check the complete current contiguous view and physical word backing."""
    count = t.numel()
    if t.storage_offset() < 0 or 4 * (t.storage_offset() + count) > t.untyped_storage().nbytes():
        raise ValueError(f"{name}: storage extent")
    if count and t.data_ptr() % 4:
        raise ValueError(f"{name}: four-byte alignment")
    if t.untyped_storage().data_ptr() + t.untyped_storage().nbytes() > 2**64:
        raise ValueError(f"{name}: pointer backing")
    if t.data_ptr() + 4 * count > 2**64:
        raise ValueError(f"{name}: complete word pointer bounds")


def _validate(q, k, v, rows, cols, row_stride, col_stride, layouts, scale,
              heads, kv_heads, length, bm, even_m, bn, even_n, bd, blocks):
    parameters = (row_stride, col_stride, layouts, heads, kv_heads, blocks)
    if any(type(n) is not int or n <= 0 or n >= 2**31 for n in parameters):
        raise ValueError("positive signed32 dimensions and CSR layout strides required")
    if type(length) is not int or length < 0 or length >= 2**31:
        raise ValueError("nonnegative signed32 sequence length required")
    if any(type(n) is not int or n != 16 for n in (bm, bn, bd)):
        raise ValueError("selected 16-by-16-by-16 tile specialization required")
    if type(even_m) is not bool or type(even_n) is not bool:
        raise ValueError("boolean mask flags required")
    if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("finite FP32-range scale required")
    if heads % kv_heads:
        raise ValueError("divisible grouped heads required")
    for t, h in ((q, heads), (k, kv_heads), (v, kv_heads)):
        if not isinstance(t, torch.Tensor) or t.ndim != 4 or t.dtype != torch.float32:
            raise ValueError("rank-four FP32 Q/K/V required")
        if not t.is_cuda or t.device != q.device or t.requires_grad or t.is_conj() or t.is_neg() or not t.is_contiguous():
            raise ValueError("contiguous CUDA inference tensors on one device required")
        if t.shape[0] != q.shape[0] or t.shape[1] != h or t.shape[2] < length or t.shape[3] != bd * blocks:
            raise ValueError("Q/K/V shape or complete channel decomposition mismatch")
        if any(n < 0 or n >= 2**31 for n in t.shape):
            raise ValueError("signed32 shapes required")
    for t in (rows, cols):
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or not t.is_cuda or t.device != q.device:
            raise ValueError("CUDA int32 CSR arrays on the input device required")
        if not t.is_contiguous() or t.is_conj() or t.is_neg() or t.numel() >= 2**31:
            raise ValueError("materialized contiguous signed32 CSR storage required")
    if q.shape[0] == 0 or length == 0:
        return False
    if q.shape[0] * heads > 65535 or blocks > 65535:
        raise ValueError("CUDA grid axis bound")
    for t, name in ((q, "Q"), (k, "K"), (v, "V"), (rows, "CSR rows"), (cols, "CSR cols")):
        _storage(t, name)
    for t in (q, k, v):
        if any(s <= 0 or s >= 2**31 for s in t.stride()):
            raise ValueError("positive signed32 strides required")
        formed = sum((n - 1) * st for n, st in zip(t.shape, t.stride())) + (bm + bn) * t.stride(2)
        if formed >= 2**31:
            raise ValueError("formed tensor offsets must fit signed32")
    if even_m and q.shape[2] < triton.cdiv(length, bm) * bm:
        raise ValueError("unchecked Q loads require allocated row padding")
    if (q.shape[0] * heads * length + bm) * bd * blocks >= 2**31:
        raise ValueError("formed output offsets must fit signed32")
    row_count = triton.cdiv(length, bm)
    if (layouts - 1) * row_stride + row_count + 1 > rows.numel():
        raise ValueError("CSR row endpoint allocation")
    if layouts * col_stride > cols.numel():
        raise ValueError("CSR column layout allocation")
    # Host inspection is confined to signed integer metadata. Decreasing row
    # endpoints mean no visits; duplicates preserve their source multiplicity.
    row_values = rows.detach().reshape(-1).cpu().tolist()
    col_values = cols.detach().reshape(-1).cpu().tolist()
    for layout in range(min(layouts, heads)):
        ends = row_values[layout * row_stride:layout * row_stride + row_count + 1]
        if any(first < 0 or first > col_stride for first in ends):
            raise ValueError("CSR row endpoint outside its column layout")
        for first, last in zip(ends, ends[1:]):
            for slot in range(first, last):
                col = col_values[layout * col_stride + slot]
                if col < 0 or col >= triton.cdiv(length, bn):
                    raise ValueError("visited CSR column outside sequence tile range")
                if even_n and (col + 1) * bn > min(k.shape[2], v.shape[2]):
                    raise ValueError("unchecked K/V loads require allocated row padding")
    return True


def _output_storage(out, inputs):
    _storage(out, "output")
    lo, hi = out.data_ptr(), out.data_ptr() + 4 * out.numel()
    for t in inputs:
        base, size = t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()
        if size and not (hi <= base or base + size <= lo):
            raise ValueError("fresh output overlaps an input view")

# This kernel is for prompt only and assume that past sequence length is 0. It only supports right padding.
@triton.jit
def block_sparse_attention_kernel(
    out,  # output [B, H, M, D]. Note that B is batch_size, H is num_heads, M is q_seq_len, and D is head_size
    Q,  # query [B, H, M, D]
    K,  # key [B, H_kv, N, D]. Note that N is max_seq_len for kv cache, H_kv is num_kv_heads
    V,  # value [B, H_kv, N, D]
    layout_csr_row_indices,  # block mask CSR format. Shape is [L, num_rows + 1] where num_rows = max_seq_len / BLOCK_M
    layout_csr_col_indices,  # block mask CSR format. Shape is [L, num_rows * num_cols] where num_cols = max_seq_len / BLOCK_N
    layout_csr_row_stride_h,  # stride per head for csr_row_indices, i.e. num_rows + 1
    layout_csr_col_stride_h,  # stride per head for csr_col_indices, i.e. num_rows * num_cols
    num_layout,  # number of sparse layout (L)
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    num_heads,
    num_kv_heads,
    total_seq_len,  # Total sequence length including past sequence length and query sequence length.
    BLOCK_M: tl.constexpr,  # block size for q_seq_len
    EVEN_M: tl.constexpr,  # whether q_seq_len % BLOCK_M == 0
    BLOCK_N: tl.constexpr,  # block size for k_seq_len
    EVEN_N: tl.constexpr,  # whether k_seq_len % BLOCK_N == 0
    BLOCK_D: tl.constexpr,  # block size for D
    NUM_D_BLOCKS: tl.constexpr,  # number of data blocks =  D / BLOCK_D
):

    # Past sequence length is 0 since this kernel is for prompt only.
    q_seq_len = total_seq_len

    # Grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads, NUM_D_BLOCKS]
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    channel_block = tl.program_id(2)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads

    # For group query attention, map the query head index to the corresponding one for key and value.
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups

    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h_kv * stride_kh
    V += off_b * stride_vb + off_h_kv * stride_vh

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd  # [BLOCK_M, BLOCK_D]
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd  # [BLOCK_D, BLOCK_N]
    off_v = offs_n[:, None] * stride_vn + (channel_block * BLOCK_D + offs_d[None, :]) * stride_vd  # [BLOCK_N, BLOCK_D]

    # Initialize pointers to query, key, value
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    layout_h = off_h % num_layout

    # This assumes that past sequence length is 0, otherwise need + (past_seq_len + 1) // BLOCK_M.
    layout_ptr = layout_csr_row_indices + layout_h * layout_csr_row_stride_h + start_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + 1).to(tl.int32)

    # Loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # Fold every channel block; the third grid axis owns one output block.
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for score_block in tl.static_range(NUM_D_BLOCKS):
            if EVEN_M:
                q = tl.load(q_ptrs + score_block * BLOCK_D * stride_qd)
            else:
                q = tl.load(q_ptrs + score_block * BLOCK_D * stride_qd,
                            mask=offs_m[:, None] < q_seq_len, other=0)
            if EVEN_N:
                k = tl.load(k_ptrs + start_n * stride_kn + score_block * BLOCK_D * stride_kd)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn + score_block * BLOCK_D * stride_kd,
                            mask=offs_n[None, :] + start_n < total_seq_len, other=0)
            qk += tl.dot(q, k)

        qk *= softmax_scale

        # This assumes that past sequence length is 0, otherwise need offs_m[:, None] + past_seq_len >= ...
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)
        # update acc
        if EVEN_N:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < total_seq_len, other=0)
        acc += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_o = off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + channel_block * BLOCK_D + offs_d[None, :]
    out_ptrs = out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)

# Define the wrapper function to call the Triton kernel
def block_sparse_attention(
    Q, K, V, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h,
    num_layout, softmax_scale, num_heads, num_kv_heads, total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS
):
    launch = _validate(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
              layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout,
              softmax_scale, num_heads, num_kv_heads, total_seq_len,
              BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS)
    softmax_scale = float(softmax_scale)
    # Determine the grid size
    q_seq_len = total_seq_len
    grid = (triton.cdiv(q_seq_len, BLOCK_M), Q.shape[0] * num_heads, NUM_D_BLOCKS)

    # Allocate output tensor
    out = torch.empty((Q.shape[0], num_heads, q_seq_len, Q.shape[-1]), device=Q.device, dtype=Q.dtype)

    if not launch:
        return out
    _output_storage(out, (Q, K, V, layout_csr_row_indices, layout_csr_col_indices))

    # Call the Triton kernel
    with torch.cuda.device(Q.device):
        block_sparse_attention_kernel[grid](
            out, Q, K, V, layout_csr_row_indices, layout_csr_col_indices, layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, softmax_scale, Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3), K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3), out.stride(0), out.stride(1), out.stride(2), num_heads, num_kv_heads,
            total_seq_len, BLOCK_M=BLOCK_M, EVEN_M=EVEN_M, BLOCK_N=BLOCK_N, EVEN_N=EVEN_N, BLOCK_D=BLOCK_D, NUM_D_BLOCKS=NUM_D_BLOCKS, num_warps=4, num_stages=3
        )
    return out

import torch

import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_repaired(
    Q,
    K,
    V,
    sm_scale,
    B_Start_Loc,
    B_Seqlen,  # B_LOC records the input positions for each batch; B_SEQ_len records the actual input lengths.
    Out,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    sliding_window,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    start_m = tl.program_id(2).to(tl.int64)

    cur_kv_head = cur_head // kv_group_num

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)

    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(
                k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
                other=0.0,
            )
            
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale
            # [SYM] Mask outside the windows; use a large negative value instead of -inf.
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, -1e9)
            qk = tl.where((offs_m[:, None] - (start_n + offs_n[None, :])) < sliding_window, qk, -1e9)

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            # Avoid numerical issues when m_ij is -1e9.
            m_ij = tl.where(m_ij == -1e9, 0.0, m_ij)
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, 1)
            
            # -- update m_i and l_i
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            beta = tl.exp(m_ij - m_i_new)
            l_i_new = alpha * l_i + beta * l_ij
            l_i_new = tl.where(l_i_new == 0.0, 1e-9, l_i_new)  # Avoid division by zero.
            
            # -- update output accumulator --
            # scale p
            p_scale = beta / l_i_new
            p = p * p_scale[:, None]
            # scale acc
            acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(
                v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
                other=0.0,
            )

            p = p.to(v.dtype)
            acc += tl.dot(p, v)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return


def _checked_arguments(q, k, v, o, b_start_loc, b_seq_len, max_input_len, sliding_window):
    tensors = (q, k, v, o, b_start_loc, b_seq_len)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all buffers must be tensors")
    if any(t.ndim != 3 for t in (q, k, v, o)) or b_start_loc.ndim != 1 or b_seq_len.ndim != 1:
        raise ValueError("expected packed rank-three Q/K/V/O and metadata vectors")
    tokens, heads, dim = q.shape
    kvheads = k.shape[1]
    if dim not in (16, 32, 64, 128) or k.shape != (tokens, kvheads, dim) or v.shape != k.shape or o.shape != q.shape:
        raise ValueError("matching packed token counts, head width and output shape required")
    if heads <= 0 or kvheads <= 0 or heads % kvheads:
        raise ValueError("positive Hq divisible by positive Hkv required")
    if b_start_loc.shape != b_seq_len.shape:
        raise ValueError("metadata vector sizes must match")
    if type(max_input_len) is not int or max_input_len < 0 or type(sliding_window) is not int:
        raise ValueError("maximum length and window must be Python integers; maximum length must be nonnegative")
    if any(t.dtype != torch.float32 for t in (q, k, v, o)):
        raise TypeError("FP32 Q/K/V/O required")
    if any(t.dtype != torch.int32 for t in (b_start_loc, b_seq_len)):
        raise TypeError("signed-int32 starts and lengths required")
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("all buffers must share one CUDA device")
    if any(t.is_neg() or not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors without lazy-negative views required")
    starts, lengths = b_start_loc.tolist(), b_seq_len.tolist()
    intervals = []
    for start, length in zip(starts, lengths):
        if start < 0 or length < 0 or start+length > tokens:
            raise ValueError("packed sequence interval is out of bounds")
        if length:
            intervals.append((start, start+length))
    if any(a[0] < b[1] and b[0] < a[1] for i, a in enumerate(intervals) for b in intervals[i+1:]):
        raise ValueError("packed sequence intervals must not overlap")
    blocks = triton.cdiv(max_input_len, 128)
    if not starts or not blocks:
        return False
    if len(starts) >= 2**31 or heads > 65535 or blocks > 65535:
        raise ValueError("grid axis exceeds the CUDA launch bound")
    if heads*dim >= 2**63 or kvheads*dim >= 2**63:
        raise ValueError("dense strides exceed signed-int64 ABI")
    for t in tensors:
        if t.data_ptr()+4*t.numel() > 2**64:
            raise ValueError("buffer exceeds the native pointer address space")
        if t.numel() and t.data_ptr()%4:
            raise ValueError("nonempty buffers require four-byte alignment")
    lo, hi = o.data_ptr(), o.data_ptr()+4*o.numel()
    for t in (q, k, v, b_start_loc, b_seq_len):
        start, end = t.data_ptr(), t.data_ptr()+4*t.numel()
        if not (hi <= start or end <= lo):
            raise ValueError("output and input view byte intervals must be disjoint")
    return True


@torch.no_grad()
def context_attention_fwd_repaired(q, k, v, o, b_start_loc, b_seq_len, max_input_len, sliding_window):
    """Checked packed FP32 Mistral context attention; mutates o, returns None."""
    if not _checked_arguments(q, k, v, o, b_start_loc, b_seq_len, max_input_len, sliding_window):
        return
    dim, heads, kvheads = q.shape[2], q.shape[1], k.shape[1]
    window = min(max(sliding_window, 0), (1 << 63)-1)
    with torch.cuda.device(q.device):
        _fwd_kernel_repaired[(b_seq_len.numel(), heads, triton.cdiv(max_input_len, 128))](
            q, k, v, 1.0/(dim**0.5), b_start_loc, b_seq_len, o,
            heads*dim, dim, 1, kvheads*dim, dim, 1,
            kvheads*dim, dim, 1, heads*dim, dim, 1,
            kv_group_num=heads//kvheads, sliding_window=window,
            BLOCK_M=128, BLOCK_DMODEL=dim, BLOCK_N=128,
            num_warps=4 if dim <= 64 else 8, num_stages=1,
        )

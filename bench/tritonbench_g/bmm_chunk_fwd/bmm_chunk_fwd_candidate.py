
import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'K', 'IS_CAUSAL'],
)
@triton.jit
def _bmm_chunk_fwd_kernel_repaired(
    a_ptr, b_ptr, out_ptr, seq_idx_ptr,
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_bk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm, stride_outn,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    seqlen = tl.cast(seqlen, tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    K = tl.cast(K, tl.int64)
    ngroups = tl.cast(ngroups, tl.int64)
    stride_a_batch = tl.cast(stride_a_batch, tl.int64)
    stride_a_seqlen = tl.cast(stride_a_seqlen, tl.int64)
    stride_a_head = tl.cast(stride_a_head, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_b_batch = tl.cast(stride_b_batch, tl.int64)
    stride_b_seqlen = tl.cast(stride_b_seqlen, tl.int64)
    stride_b_head = tl.cast(stride_b_head, tl.int64)
    stride_bk = tl.cast(stride_bk, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_chunk = tl.cast(stride_out_chunk, tl.int64)
    stride_out_head = tl.cast(stride_out_head, tl.int64)
    stride_outm = tl.cast(stride_outm, tl.int64)
    stride_outn = tl.cast(stride_outn, tl.int64)
    stride_seq_idx_batch = tl.cast(stride_seq_idx_batch, tl.int64)
    stride_seq_idx_seqlen = tl.cast(stride_seq_idx_seqlen, tl.int64)
    pid_b = tl.program_id(axis=1).to(tl.int64)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - ngroups * pid_c
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0).to(tl.int64) // num_pid_n
    pid_n = tl.program_id(axis=0).to(tl.int64) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(dot_dtype)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))

@triton.jit
def _clear_causal_upper(Out, chunk_size, chunk_heads, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    batch = tl.program_id(1).to(tl.int64)
    chunk_head = tl.program_id(2).to(tl.int64)
    chunk_size = tl.cast(chunk_size, tl.int64)
    chunk_heads = tl.cast(chunk_heads, tl.int64)
    tiles_n = tl.cdiv(chunk_size, BLOCK_N)
    row = pid // tiles_n * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    column = pid % tiles_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offsets = (batch * chunk_heads + chunk_head) * chunk_size * chunk_size + row[:, None] * chunk_size + column[None, :]
    tl.store(Out + offsets, 0.0, mask=(row[:, None] < chunk_size) & (column[None, :] < chunk_size) & (column[None, :] > row[:, None]))


def _bmm_chunk_fwd_repaired(a, b, chunk_size, seq_idx=None, causal=False, output_dtype=None):
    tensors = (a, b) if seq_idx is None else (a, b, seq_idx)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("tensor inputs required")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided inputs required")
    if a.dtype != torch.float32 or b.dtype != torch.float32 or (seq_idx is not None and seq_idx.dtype != torch.int64):
        raise TypeError("FP32 A/B and INT64 sequence IDs required")
    if output_dtype is not None and output_dtype != torch.float32:
        raise TypeError("FP32 output required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if any(t.device != a.device for t in tensors):
        raise ValueError("one CUDA device required")
    if a.ndim not in (3, 4) or b.shape != a.shape:
        raise ValueError("A/B must have matching rank-3 or rank-4 shapes")
    if seq_idx is not None and seq_idx.shape != a.shape[:2]:
        raise ValueError("sequence IDs must have shape [batch, sequence]")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous inputs required")
    grouped = a.ndim == 4
    expected = (a.shape[1] * (a.shape[2] if grouped else 1) * a.shape[-1],
                (a.shape[2] if grouped else 1) * a.shape[-1],
                a.shape[-1] if grouped else 0, 1)
    if any(t.stride(-1) != 1 or (t.numel() and
               (t.stride(0), t.stride(1), t.stride(2) if grouped else 0, t.stride(-1)) != expected)
           for t in (a, b)) or (seq_idx is not None and seq_idx.numel() and
                               (seq_idx.stride(0), seq_idx.stride(1)) != (a.shape[1], 1)):
        raise ValueError("literal contiguous source strides required")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views are unsupported")
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("positive integer chunk size required")
    if type(causal) is not bool:
        raise TypeError("causal must be Boolean")
    has_groups = a.ndim == 4
    batch, seqlen = a.shape[:2]
    ngroups, k = (a.shape[2], a.shape[3]) if has_groups else (1, a.shape[2])
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    output_shape = ((batch, nchunks, ngroups, chunk_size, chunk_size) if has_groups else
                    (batch, nchunks, chunk_size, chunk_size))
    count = batch * nchunks * ngroups * chunk_size * chunk_size
    if any(t.numel() * t.element_size() >= (1 << 63) for t in tensors) or 4 * count >= (1 << 63):
        raise ValueError("byte extents must fit signed int64")
    if count and (batch >= (1 << 16) or nchunks * ngroups >= (1 << 16) or
                  any(triton.cdiv(chunk_size, m) * triton.cdiv(chunk_size, n) >= (1 << 31)
                      for m, n in ((128,256),(64,256),(128,128),(128,64),(64,128),(128,32),(64,32),(32,64),(64,64)))):
        raise ValueError("launch dimensions exceed CUDA limits")
    if any(t.numel() and t.data_ptr() % t.element_size() for t in tensors):
        raise ValueError("naturally aligned inputs required")
    if any(not (x.data_ptr() + x.numel() * x.element_size() <= y.data_ptr() or
                y.data_ptr() + y.numel() * y.element_size() <= x.data_ptr())
           for i, x in enumerate(tensors) for y in tensors[i + 1:]):
        raise ValueError("input data spans must be disjoint")
    # Retain the original intentional conditional layout operations.
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty(output_shape, device=a.device, dtype=out_dtype)
    if count == 0:
        return out
    grid = lambda meta: (triton.cdiv(chunk_size, meta['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, meta['BLOCK_SIZE_N']),
                         batch, nchunks * ngroups)
    with torch.cuda.device(a.device):
        _bmm_chunk_fwd_kernel_repaired[grid](
            a, b, out, seq_idx, seqlen, chunk_size, k, ngroups,
            a.stride(0), a.stride(1), a.stride(2) if has_groups else 0, a.stride(-1),
            b.stride(0), b.stride(1), b.stride(2) if has_groups else 0, b.stride(-1),
            out.stride(0), out.stride(1), out.stride(2) if has_groups else 0, out.stride(-2), out.stride(-1),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            causal, tl.float32, HAS_SEQ_IDX=seq_idx is not None)
        if causal:
            _clear_causal_upper[(triton.cdiv(chunk_size, 32) * triton.cdiv(chunk_size, 64), batch, nchunks * ngroups)](
                out, chunk_size, nchunks * ngroups, BLOCK_M=32, BLOCK_N=64)
    return out

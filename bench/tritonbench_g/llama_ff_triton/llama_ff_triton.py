
import torch
import triton
import triton.language as tl

@triton.jit
def ff_llama(
    a_ptr, w1_ptr, w3_ptr, out_ptr, rms_w_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    stride_rms_w,
    USE_FP8: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    w1 and w3 are weights (linear layers)
    F.silu(w1(x)) * w3(x)
    """
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_am = tl.cast(stride_am, tl.int64)
    stride_ak = tl.cast(stride_ak, tl.int64)
    stride_w1k = tl.cast(stride_w1k, tl.int64)
    stride_w1n = tl.cast(stride_w1n, tl.int64)
    stride_w3k = tl.cast(stride_w3k, tl.int64)
    stride_w3n = tl.cast(stride_w3n, tl.int64)
    stride_outm = tl.cast(stride_outm, tl.int64)
    stride_outn = tl.cast(stride_outn, tl.int64)
    stride_rms_w = tl.cast(stride_rms_w, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K)[None, :] * stride_rms_w
    a_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for step in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        a_sum += tl.extra.cuda.libdevice.pow(a.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        a = a * rms_w
        b = tl.load(w1_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            b = b.to(tl.float8e5, bitcast=True)
            b = b.to(tl.float32)
            b = b.to(tl.float16)
        acc1 += tl.dot(a, b)
        c = tl.load(w3_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            c = c.to(tl.float8e5, bitcast=True)
            c = c.to(tl.float32)
            c = c.to(tl.float16)
        acc2 += tl.dot(a, c)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w

    a_mean = tl.sum(a_sum, axis=1) / K + EPS
    a_norm = tl.math.rsqrt(a_mean)
    acc1 = acc1 * a_norm[:, None]
    acc2 = acc2 * a_norm[:, None]
    accumulator = (acc1 * tl.sigmoid(acc1)) * acc2

    offs_outm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


def kernel_ff(x: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor, rms_w: torch.Tensor) -> torch.Tensor:
    tensors = (x, w1, w3, rms_w)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("feed-forward inputs must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("feed-forward inputs must be strided")
    if x.dtype != torch.float16 or w1.dtype not in (torch.float16, torch.int8) or not w1.dtype == w3.dtype == rms_w.dtype:
        raise TypeError("X must be FP16; weights must share FP16 or int8 E5M2 storage")
    if x.ndim != 3 or w1.ndim != 2 or w3.ndim != 2 or rms_w.ndim != 1:
        raise ValueError("expected X[B,S,K], W1/W3[N,K], RMS[K]")
    if not all(t.is_cuda and t.device == x.device for t in tensors):
        raise ValueError("feed-forward inputs must share a CUDA device")
    if any(not t.is_contiguous() or t.storage_offset() != 0 or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("feed-forward inputs require contiguous zero-offset storage without lazy views")
    batch, seq_len, K = x.shape
    M = batch * seq_len
    N = w1.shape[0]
    if K == 0 or N == 0 or w1.shape != (N, K) or w3.shape != (N, K) or rms_w.shape != (K,):
        raise ValueError("positive K/N and matching weight shapes are required")
    if max(M, N, K) >= 2**31:
        raise ValueError("launch dimensions must fit signed 32-bit arguments")
    tiles_m, tiles_n = (M + 15) // 16, (N + 15) // 16
    if tiles_m * tiles_n > 2**31 - 1:
        raise ValueError("feed-forward grid exceeds the CUDA x-grid limit")
    formed_k = ((K + 63) // 64) * 64 + 63
    weight_bytes = 1 if w1.dtype == torch.int8 else 2
    if 2 * (M * K + formed_k) >= 2**63 or weight_bytes * (N * K + formed_k) >= 2**63 or 2 * (tiles_m * 16 * N + tiles_n * 16) >= 2**63:
        raise ValueError("formed source byte offsets must fit signed 64-bit arithmetic")
    if w1.dtype == torch.int8 and any(x.untyped_storage()._cdata == t.untyped_storage()._cdata for t in (w1, w3, rms_w)):
        raise ValueError("FP16 input and byte weights require distinct backing storage")
    if M and any(t.data_ptr() % 16 for t in tensors):
        raise ValueError("nonempty feed-forward pointers require 16-byte alignment")

    # These operations preserve the admitted inputs' backing storage.
    w1_t = w1.t()
    w3_t = w3.t()
    x_reshape = x.view(M, K)
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    grid = lambda META: (triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["N"], META["BLOCK_SIZE_N"]),)
    if M > 0:
        with torch.cuda.device(x.device):
            ff_llama[grid](
                x_reshape, w1_t, w3_t, out, rms_w,
                M, N, K,
                K, 1,
                1, K,
                1, K,
                N, 1,
                1,
                USE_FP8=w1_t.dtype != torch.float16,
                EPS=1e-6,
                BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=64,
                num_stages=2, num_warps=4
            )
    return out.view(batch, seq_len, N)

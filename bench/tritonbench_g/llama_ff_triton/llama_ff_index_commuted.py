"""Existing index-commuted candidate on the selected masked scalar-cast source.

The source masks, conversions and floating operation order match ff_llama.
Only output-row integer multiplication is commuted, as in the preserved candidate.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def ff_llama_index_commuted(
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

    offs_outm = BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M)
    offs_outn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (stride_outm * offs_outm[:, None] + stride_outn * offs_outn[None, :])
    out_mask = (offs_outm[:, None] < M) & (offs_outn[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


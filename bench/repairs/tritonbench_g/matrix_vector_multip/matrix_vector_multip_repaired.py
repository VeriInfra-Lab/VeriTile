import logging

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_stages=s, num_warps=w)
        for m in [32, 64, 128]
        for n in [1, 2, 4, 8]
        for s in [3, 4]
        for w in [4, 8]
    ],
    key=["M", "N"],
)
@triton.jit
def mv_contiguous_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    N = tl.cast(N, tl.uint64)
    M = tl.cast(M, tl.uint64)
    stride_an = tl.cast(stride_an, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_bm = tl.cast(stride_bm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(0).to(tl.uint64)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.uint64)[:, None]
    offset_m = tl.arange(0, BLOCK_M).to(tl.uint64)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    m = tl.full((), 0, tl.uint64)
    while m < M:
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm
        m += BLOCK_M

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


# Candidate: make the already fused native carry explicit in source.
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": m, "BLOCK_N": n}, num_stages=s, num_warps=w)
        for m in [32, 64, 128]
        for n in [1, 2, 4, 8]
        for s in [3, 4]
        for w in [4, 8]
    ],
    key=["M", "N"],
)
@triton.jit
def mv_contiguous_fma_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    N = tl.cast(N, tl.uint64)
    M = tl.cast(M, tl.uint64)
    stride_an = tl.cast(stride_an, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_bm = tl.cast(stride_bm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(0).to(tl.uint64)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.uint64)[:, None]
    offset_m = tl.arange(0, BLOCK_M).to(tl.uint64)[None, :]
    n_mask = offset_n < N
    A_ptrs = A + offset_n * stride_an + offset_m * stride_am
    B_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    m = tl.full((), 0, tl.uint64)
    while m < M:
        m_mask = m + offset_m < M
        a = tl.load(A_ptrs, mask=n_mask & m_mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc = tl.fma(a, b, acc)
        A_ptrs += BLOCK_M * stride_am
        B_ptrs += BLOCK_M * stride_bm
        m += BLOCK_M

    acc = tl.sum(acc, axis=1)
    C_ptrs = C + offset_n * stride_cn
    tl.store(C_ptrs, acc[:, None], mask=n_mask)


def mv(inp, vec):
    if not isinstance(inp, torch.Tensor) or not isinstance(vec, torch.Tensor):
        raise TypeError("inputs must be tensors")
    if inp.layout != torch.strided or vec.layout != torch.strided:
        raise ValueError("inputs must use strided storage")
    if inp.dtype != torch.float32 or vec.dtype != torch.float32:
        raise ValueError("inputs must be FP32")
    if not inp.is_cuda or not vec.is_cuda:
        raise ValueError("inputs must be CUDA tensors")
    if inp.device != vec.device:
        raise ValueError("inputs must be on the same device")
    if inp.ndim != 2 or vec.ndim != 1:
        raise ValueError("expected a matrix and a vector")
    if inp.shape[1] != vec.shape[0]:
        raise ValueError("incompatible dimensions")
    if not inp.is_contiguous() or not vec.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if inp.is_neg() or vec.is_neg():
        raise ValueError("inputs must have no unresolved negative view")
    N, M = inp.shape
    if N >= 2**63 or M >= 2**63:
        raise ValueError("dimensions must fit signed int64")
    if ((inp.numel() > 0 and inp.data_ptr() % 4 != 0)
            or (vec.numel() > 0 and vec.data_ptr() % 4 != 0)):
        raise ValueError("nonempty input pointers must be aligned to 4 bytes")
    out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch.cuda.device(inp.device):
        mv_contiguous_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            M,
            1,
            1,
            1,
        )
    return out

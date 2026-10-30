"""Authored Python counterpart of the pre-existing Lean integer-mask candidate.

This is newly materialized source, not the historical original reference.
The repaired original remains dequantize_matmul_repaired.py.
"""
import triton
import triton.language as tl

@triton.jit
def dequantize_index_commuted(
    b_ptr, b_scale_ptr, fpb_ptr,
    K, N,
    stride_bk, stride_bn, stride_scale,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0).to(tl.int64)
    n_block_idx = tl.program_id(axis=1).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_scale
    n_mask = BLOCK_SIZE_N * n_block_idx + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)

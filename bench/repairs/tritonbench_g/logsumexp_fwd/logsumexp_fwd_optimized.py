"""Candidate: remove the redundant post-load select when scaling is disabled.
All scales, original wrapper guards and the scaled nonfinite path are retained.
"""
import math
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['D']
)
@triton.heuristics({
    'HAS_SCALE': lambda args: args['scale'] is not None
})
@triton.jit
def logsumexp_fwd_kernel_optimized(
    x: tl.tensor,
    z: tl.tensor,
    scale: tl.float32,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_SCALE: tl.constexpr
):
    i_n, i_d = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    if HAS_SCALE:
        b_x = b_x * scale
        b_x = tl.where(m_d, b_x, -float('inf'))
    b_m = tl.max(b_x, 0)
    positive_inf = float('inf')
    b_m = tl.where((b_m == positive_inf) | (b_m == -float('inf')), 0.0, b_m)
    b_z = tl.log(tl.sum(tl.exp(b_x - b_m), 0)) + b_m
    tl.store(z + i_n * tl.cdiv(D, B) + i_d, b_z)

def logsumexp_fwd_optimized(x, scale: Optional[float] = None, dtype: Optional[torch.dtype] = None):
    if not isinstance(x, torch.Tensor):
        raise TypeError("input must be a tensor")
    if x.layout != torch.strided or x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError("input must be a strided FP32 CUDA tensor")
    if x.ndim == 0:
        raise ValueError("input must have a last dimension")
    if not x.is_contiguous() or x.is_neg():
        raise ValueError("input must be contiguous with no unresolved negative view")
    if x.numel() >= 2**31:
        raise ValueError("input must contain fewer than 2**31 elements")
    if x.numel() and x.data_ptr() % 4:
        raise ValueError("nonempty FP32 input must be 4-byte aligned")
    if scale is not None and not isinstance(scale, float):
        raise TypeError("scale must be a Python float or None")
    if dtype not in (None, torch.float32, torch.float64):
        raise ValueError("output dtype must be FP32 or FP64")
    shape = x.shape
    D = shape[-1]
    N = math.prod(shape[:-1])
    if N >= 2**31:
        raise ValueError("output must contain fewer than 2**31 elements")
    # Preserve the original view operation; explicit extents also handle D=0.
    x = x.view(N, D)
    with torch.cuda.device(x.device):
        if D == 0:
            z = x.new_full(shape[:-1], -float('inf'), dtype=torch.float32)
        else:
            B = min(triton.next_power_of_2(D), 64 * 1024)
            ND = triton.cdiv(D, B)
            z = x.new_empty(N, ND, dtype=torch.float32)
            if N:
                logsumexp_fwd_kernel_optimized[(N, ND)](x=x, z=z, scale=scale, D=D, B=B)
            z = z.logsumexp(-1).view(shape[:-1])
        if dtype is not None and dtype != torch.float32:
            z = z.to(dtype)
    return z

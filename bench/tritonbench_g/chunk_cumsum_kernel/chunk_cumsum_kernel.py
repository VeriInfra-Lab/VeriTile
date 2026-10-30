import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=4),
    ],
    key=[]
)
@triton.jit
def chunk_global_cumsum_scalar_kernel_repaired(
    s,
    o,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(0, tl.int64), tl.cast(tl.cdiv(T, BT), tl.int64)):
        times = i_t * BT + tl.arange(0, BT).to(tl.int64)
        p_s = s + i_bh.to(tl.int64) * T + times
        p_o = o + i_bh.to(tl.int64) * T + times
        b_s = tl.load(p_s, mask=times < T).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=times < T)

def chunk_global_cumsum_scalar_repaired(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not isinstance(s, torch.Tensor):
        raise TypeError("s must be a Tensor")
    if s.layout != torch.strided:
        raise ValueError("s must have strided layout")
    if s.dtype != torch.float32:
        raise TypeError("s must be FP32")
    if dtype is not None and dtype != torch.float32:
        raise TypeError("dtype must be None or FP32")
    if s.ndim != 3:
        raise ValueError("s must have rank three")
    if not s.is_cuda:
        raise ValueError("s must be CUDA")
    if not s.is_contiguous():
        raise ValueError("s must be contiguous")
    if s.is_neg():
        raise ValueError("s must not have an unresolved negative view")
    B, H, T = s.shape
    if s.numel() != 0 and B * H >= 2**31:
        raise ValueError("nonempty row grid must fit CUDA grid-x")
    if s.numel() != 0 and s.data_ptr() % 4 != 0:
        raise ValueError("nonempty FP32 input must be 4-byte aligned")
    dtype = torch.float32
    grid = (B * H,)
    z = torch.empty_like(s, dtype=dtype)
    if z.numel() == 0:
        return z
    with torch.cuda.device(s.device):
        chunk_global_cumsum_scalar_kernel_repaired[grid](s, z, T=T)
    return z


# Intended sum-left candidate; production dispatch remains above.
@triton.jit
def chunk_global_cumsum_scalar_kernel_sum_left(
    s,
    o,
    T: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(0, tl.int64), tl.cast(tl.cdiv(T, BT), tl.int64)):
        times = i_t * BT + tl.arange(0, BT).to(tl.int64)
        p_s = s + i_bh.to(tl.int64) * T + times
        p_o = o + i_bh.to(tl.int64) * T + times
        b_s = tl.load(p_s, mask=times < T).to(tl.float32)
        b_o = tl.cumsum(b_s, axis=0) + b_z[None]
        b_zz = tl.sum(b_s, axis=0)
        b_z = b_zz + b_z
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=times < T)


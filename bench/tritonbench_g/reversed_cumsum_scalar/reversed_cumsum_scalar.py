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
def chunk_global_reversed_cumsum_scalar_kernel(
    s,
    o,
    T: tl.constexpr,
    H: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_st: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ot: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes = tl.arange(0, BT).to(tl.int64)
        active = lanes < T - start_t
        times = start_t + tl.where(active, lanes, 0)
        p_s = s + (i_bh // H) * stride_sb + (i_bh % H) * stride_sh + times * stride_st
        p_o = o + (i_bh // H) * stride_ob + (i_bh % H) * stride_oh + times * stride_ot
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=active)


def chunk_global_reversed_cumsum_scalar(
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
        chunk_global_reversed_cumsum_scalar_kernel[grid](
            s, z,
            T=T, H=H,
            stride_sb=s.stride(0), stride_sh=s.stride(1), stride_st=s.stride(2),
            stride_ob=z.stride(0), stride_oh=z.stride(1), stride_ot=z.stride(2),
        )
    return z


# The intended kernel-equivalence candidate; wrapper selection is unchanged.
@triton.jit
def chunk_global_reversed_cumsum_scalar_kernel_index_commuted(
    s,
    o,
    T: tl.constexpr,
    H: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_st: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_ot: tl.constexpr,
    BT: tl.constexpr,
):
    i_bh = tl.program_id(0).to(tl.int64)
    b_z = tl.zeros([], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes = tl.arange(0, BT).to(tl.int64)
        active = lanes < T - start_t
        times = start_t + tl.where(active, lanes, 0)
        p_s = s + stride_sb * (i_bh // H) + stride_sh * (i_bh % H) + times * stride_st
        p_o = o + (i_bh // H) * stride_ob + (i_bh % H) * stride_oh + times * stride_ot
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_zz = tl.sum(b_s, axis=0)
        b_z += b_zz
        b_o = b_s - tl.cumsum(b_s, axis=0) + b_z[None]
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=active)

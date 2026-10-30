
import torch
import triton
import triton.language as tl
from typing import Optional

@triton.autotune(
    configs=[
        triton.Config({'BT': 16}, num_warps=2),
        triton.Config({'BT': 16}, num_warps=4),
        triton.Config({'BT': 16}, num_warps=8),
        triton.Config({'BT': 32}, num_warps=2),
        triton.Config({'BT': 32}, num_warps=4),
        triton.Config({'BT': 32}, num_warps=8),
        triton.Config({'BT': 64}, num_warps=2),
        triton.Config({'BT': 64}, num_warps=4),
        triton.Config({'BT': 64}, num_warps=8),
    ],
    key=['S']
)
@triton.jit
def chunk_global_cumsum_vector_kernel(
    s,
    z,
    H: tl.constexpr,
    s_s_b: tl.constexpr,
    s_s_h: tl.constexpr,
    s_s_t: tl.constexpr,
    s_s_d: tl.constexpr,
    s_z_b: tl.constexpr,
    s_z_h: tl.constexpr,
    s_z_t: tl.constexpr,
    s_z_d: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    b_z = tl.zeros([BS], dtype=tl.float32)
    start_s = i_s * BS
    lanes_s = tl.arange(0, BS).to(tl.int64)
    active_s = lanes_s < S - start_s
    columns = start_s + tl.where(active_s, lanes_s, 0)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64)):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes_t = tl.arange(0, BT).to(tl.int64)
        active_t = lanes_t < T - start_t
        times = start_t + tl.where(active_t, lanes_t, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_b * s_s_b + i_h * s_s_h + times[:, None] * s_s_t + columns[None, :] * s_s_d
        p_z = z + i_b * s_z_b + i_h * s_z_h + times[:, None] * s_z_t + columns[None, :] * s_z_d
        # [BT, BS]; zero inactive lanes before the triangular dot.
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)
        if i_t >= 0:
            b_z += tl.sum(b_s, 0)

def chunk_global_cumsum_vector(
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
    if s.ndim != 4:
        raise ValueError("s must have rank four")
    if not s.is_cuda:
        raise ValueError("s must be CUDA")
    if not s.is_contiguous():
        raise ValueError("s must be contiguous")
    if s.is_neg():
        raise ValueError("s must not have an unresolved negative view")
    B, H, T, S = s.shape
    if s.numel() != 0 and (B * H > 65535 or triton.cdiv(S, 32) >= 2**31):
        raise ValueError("nonempty grid must fit CUDA grid-x/y")
    if s.numel() != 0 and s.data_ptr() % 4 != 0:
        raise ValueError("nonempty FP32 input must be 4-byte aligned")
    dtype = torch.float32
    BS = 32
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    if z.numel() == 0:
        return z
    with torch.cuda.device(s.device):
        chunk_global_cumsum_vector_kernel[grid](
            s, z,
            H=H,
            s_s_b=s.stride(0), s_s_h=s.stride(1), s_s_t=s.stride(2), s_s_d=s.stride(3),
            s_z_b=z.stride(0), s_z_h=z.stride(1), s_z_t=z.stride(2), s_z_d=z.stride(3),
            T=T, S=S, BS=BS
        )
    return z


# Intended sum-left candidate; production dispatch remains above.
@triton.jit
def chunk_global_cumsum_vector_kernel_sum_left(
    s,
    z,
    H: tl.constexpr,
    s_s_b: tl.constexpr,
    s_s_h: tl.constexpr,
    s_s_t: tl.constexpr,
    s_s_d: tl.constexpr,
    s_z_b: tl.constexpr,
    s_z_h: tl.constexpr,
    s_z_t: tl.constexpr,
    s_z_d: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] >= o_i[None, :], 1., 0.)
    b_z = tl.zeros([BS], dtype=tl.float32)
    start_s = i_s * BS
    lanes_s = tl.arange(0, BS).to(tl.int64)
    active_s = lanes_s < S - start_s
    columns = start_s + tl.where(active_s, lanes_s, 0)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64)):
        start_t = tl.cast(i_t, tl.int64) * BT
        lanes_t = tl.arange(0, BT).to(tl.int64)
        active_t = lanes_t < T - start_t
        times = start_t + tl.where(active_t, lanes_t, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_b * s_s_b + i_h * s_s_h + times[:, None] * s_s_t + columns[None, :] * s_s_d
        p_z = z + i_b * s_z_b + i_h * s_z_h + times[:, None] * s_z_t + columns[None, :] * s_z_d
        # [BT, BS]; zero inactive lanes before the triangular dot.
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)
        if i_t >= 0:
            b_z = tl.sum(b_s, 0) + b_z


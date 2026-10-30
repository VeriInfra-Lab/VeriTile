
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
def chunk_global_reversed_cumsum_vector_kernel(
    s,
    z,
    s_s_h: tl.constexpr,
    s_s_t: tl.constexpr,
    s_s_d: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[:, None] <= o_i[None, :], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        start_s = i_s * BS
        lane_t = tl.arange(0, BT).to(tl.int64)
        lane_s = tl.arange(0, BS).to(tl.int64)
        active_t = lane_t < T - start_t
        active_s = lane_s < S - start_s
        times = start_t + tl.where(active_t, lane_t, 0)
        features = start_s + tl.where(active_s, lane_s, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        p_z = z + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        # [BT, BS]
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)

def chunk_global_reversed_cumsum_vector(
    s: torch.Tensor,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not isinstance(s, torch.Tensor):
        raise TypeError("s must be a Tensor")
    if s.layout != torch.strided:
        raise ValueError("s must use strided layout")
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
        raise ValueError("s must have no unresolved negative view bit")
    if s.numel() > 0 and (s.shape[0] * s.shape[1] >= 2**31 or (s.shape[3] + 31) // 32 >= 2**31):
        raise ValueError("each grid axis must fit signed int32")
    if s.numel() > 0 and s.data_ptr() % 4 != 0:
        raise ValueError("s must have a 4-byte-aligned data pointer")
    B, H, T, S = s.shape
    BS = 32
    dtype = dtype or s.dtype
    grid = (triton.cdiv(S, BS), B * H)
    z = torch.empty_like(s, dtype=dtype)
    if z.numel() == 0:
        return z
    with torch.cuda.device(s.device):
        chunk_global_reversed_cumsum_vector_kernel[grid](
            s, z,
            T * S, S, 1,
            T=T, S=S, BS=BS
        )
    return z


# The intended kernel-equivalence candidate; wrapper selection is unchanged.
@triton.jit
def chunk_global_reversed_cumsum_vector_kernel_compare_right(
    s,
    z,
    s_s_h: tl.constexpr,
    s_s_t: tl.constexpr,
    s_s_d: tl.constexpr,
    T: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr
):
    i_s, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    o_i = tl.arange(0, BT)
    m_s = tl.where(o_i[None, :] >= o_i[:, None], 1., 0.)

    b_z = tl.zeros([BS], dtype=tl.float32)
    for i_t in range(tl.cast(tl.cdiv(T, BT), tl.int64) - 1, -1, -1):
        start_t = tl.cast(i_t, tl.int64) * BT
        start_s = i_s * BS
        lane_t = tl.arange(0, BT).to(tl.int64)
        lane_s = tl.arange(0, BS).to(tl.int64)
        active_t = lane_t < T - start_t
        active_s = lane_s < S - start_s
        times = start_t + tl.where(active_t, lane_t, 0)
        features = start_s + tl.where(active_s, lane_s, 0)
        active = active_t[:, None] & active_s[None, :]
        p_s = s + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        p_z = z + i_bh * s_s_h + times[:, None] * s_s_t + features[None, :] * s_s_d
        # [BT, BS]
        b_s = tl.load(p_s, mask=active, other=0).to(tl.float32)
        b_c = b_z[None, :] + tl.dot(m_s, b_s, allow_tf32=False)
        tl.store(p_z, b_c.to(p_z.dtype.element_ty), mask=active)

        if i_t >= 0:
            b_z += tl.sum(b_s, 0)

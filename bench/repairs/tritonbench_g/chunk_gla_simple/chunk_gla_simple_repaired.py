"""Checked direct-input FP32 candidate for chunk_gla_simple.

Original SHA256: 4822daca85e4b72d0aef46583ec015901e63a783739035caecc4df273c157ba8
The production arithmetic and original positive-K BK/BV/config choices are preserved.
No top-level CUDA demonstration is executed by importing this candidate.
"""

import math

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o_repaired(
    q,
    k,
    v,
    h,
    g,
    o,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    # Kernel implementation
    # Widen before every address product; block-pointer offsets are only i32.
    i_v = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    s_k_h, s_k_t = tl.cast(s_k_h, tl.int64), tl.cast(s_k_t, tl.int64)
    s_v_h, s_v_t = tl.cast(s_v_h, tl.int64), tl.cast(s_v_t, tl.int64)
    s_h_h, s_h_t = tl.cast(s_h_h, tl.int64), tl.cast(s_h_t, tl.int64)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    rows = i_t * BT + tl.arange(0, BT).to(tl.int64)
    cols = i_v * BV + tl.arange(0, BV).to(tl.int64)
    # Keep range's signed induction bound valid across the uint32 constant interval.
    for i_k in range(tl.cast(tl.cdiv(K, BK), tl.int64)):
        keys = tl.cast(i_k, tl.int64) * BK + tl.arange(0, BK).to(tl.int64)
        p_q = q + i_bh * s_k_h + rows[:, None] * s_k_t + keys[None, :]
        p_k = k + i_bh * s_k_h + keys[:, None] + rows[None, :] * s_k_t
        p_h = h + i_bh * s_h_h + i_t * K * V + keys[:, None] * s_h_t + cols[None, :]
        b_q = tl.load(p_q, mask=(rows[:, None] < T) & (keys[None, :] < K), other=0)
        b_k = tl.load(p_k, mask=(keys[:, None] < K) & (rows[None, :] < T), other=0)
        b_h = tl.load(p_h, mask=(keys[:, None] < K) & (cols[None, :] < V), other=0)
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    p_g = g + i_bh * T + rows
    b_g = tl.load(p_g, mask=rows < T, other=0)
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)

    p_v = v + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    b_v = tl.load(p_v, mask=(rows[:, None] < T) & (cols[None, :] < V), other=0)
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = o + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=(rows[:, None] < T) & (cols[None, :] < V))

def _check_addresses(*tensors):
    for tensor in tensors:
        pointer = tensor.data_ptr()
        if pointer < 0 or pointer % 4 or pointer + 4*tensor.numel() >= 2**64:
            raise ValueError("invalid FP32 pointer alignment or address span")


def chunk_fwd_o_fn_repaired(h, q, k, v, g, BT, scale):
    tensors = (h, q, k, v, g)
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("history, query, key, value and gate must be tensors")
    if type(BT) is not int or BT not in (16, 32, 64, 128, 256, 512, 1024):
        raise ValueError("BT must be a power of two from 16 through 1024")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("K and V must have shape [B,H,T,D]")
    B, H, T, K, V = *k.shape, v.shape[-1]
    NT = triton.cdiv(T, BT)
    if q.shape != k.shape or v.shape[:3] != k.shape[:3]:
        raise ValueError("Q/K and V batch, head, and time dimensions must agree")
    if h.shape != (B, H, NT * K, V) or g.shape != (B, H, T):
        raise ValueError("H must be [B,H,NT*K,V] and G must be [B,H,T]")
    if any(x.layout != torch.strided or x.dtype != torch.float32 or not x.is_cuda
           or x.device != k.device or not x.is_contiguous() or x.is_neg() or x.is_conj() for x in tensors):
        raise ValueError("ordinary contiguous FP32 tensors on one CUDA device are required")
    if type(scale) not in (float, int):
        raise TypeError("scale must be a builtin scalar")
    try:
        scale = float(scale)
    except OverflowError as error:
        raise ValueError("scale must be finite and within FP32 range") from error
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("scale must be finite and within FP32 range")
    if B*H*T*V:
        BK = min(triton.next_power_of_2(max(K, 1)), 64)
        BV = min(triton.next_power_of_2(V), 64)
        NV = triton.cdiv(V, BV)
        if max(4*B*H*T*K, 4*B*H*T*V, 4*B*H*NT*K*V, 4*B*H*T) >= 2**63:
            raise ValueError("storage byte extent exceeds the signed 64-bit contract")
        if NV >= 2**31 or NT > 65535 or B*H > 65535:
            raise ValueError("launch grid exceeds CUDA axis limits")
        _check_addresses(*tensors)
    o = torch.empty_like(v)
    if B*H*T*V == 0:
        return o
    _check_addresses(o)
    with torch.cuda.device(k.device):
        chunk_simple_gla_fwd_kernel_o_repaired[(NV, NT, B*H)](
            q, k, v, h, g, o,
            T*K, K, T*V, V, NT*K*V, V, scale,
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
        )
    return o

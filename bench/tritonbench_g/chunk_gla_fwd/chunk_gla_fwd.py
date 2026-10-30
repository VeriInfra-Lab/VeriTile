
import math

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC", "BK"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_i, i_j = i_c // NC, i_c % NC
    if i_t * BT + i_i * BC >= T:
        return
    if i_i <= i_j:
        return

    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gk = tl.make_block_ptr(g + i_bh * s_k_h, (K, T), (1, s_k_t), (i_k * BK, i_t * BT + i_j * BC), (BK, BC), (0, 1))
        p_gn = g + i_bh * s_k_h + (i_t * BT + i_i * BC) * K + o_k
        b_gn = tl.load(p_gn, mask=m_k, other=0)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_qg = b_q * tl.exp(b_g - b_gn[None, :]) * scale
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_gk = tl.load(p_gk, boundary_check=(0, 1), padding_option="zero")
        b_kg = b_k * tl.exp(b_gn[:, None] - b_gk)
        b_A += tl.dot(b_qg, b_kg)

    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_i * BC, i_j * BC), (BC, BC), (1, 0))
    tl.store(p_A, b_A.to(A.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BT"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr
):
    i_t, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_j = i_i
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = tl.arange(0, BK)
    o_A = i_bh * T * BT + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BT + i_j * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, 0), (BC, BK), (1, 0))

    p_k = k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k
    p_gk = g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A = tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC", "BK"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q,
    k,
    g,
    A,
    s_k_h,
    s_k_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr
):
    i_k, i_tc, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_t, i_i = i_tc // NC, i_tc % NC
    i_j = i_i
    n_bh = tl.num_programs(2)
    if i_t * BT + i_i * BC >= T:
        return

    o_i = tl.arange(0, BC)
    o_k = i_k * BK + tl.arange(0, BK)
    o_A = (i_bh + i_k * n_bh) * T * BC + (i_t * BT + i_i * BC + tl.arange(0, BC)) * BC
    m_k = o_k < K
    m_A = (i_t * BT + i_i * BC + tl.arange(0, BC)) < T

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT + i_i * BC, i_k * BK), (BC, BK), (1, 0))
    p_k = k + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k
    p_gk = g + i_bh * s_k_h + (i_t * BT + i_j * BC) * K + o_k

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
    for j in range(0, min(BC, T-i_t*BT-i_i*BC)):
        b_A = tl.zeros([BC], dtype=tl.float32)
        b_k = tl.load(p_k, mask=m_k, other=0).to(tl.float32)
        b_gk = tl.load(p_gk, mask=m_k, other=0).to(tl.float32)
        b_A += tl.sum(b_q * b_k[None, :] * tl.exp(b_g - b_gk[None, :]), 1)
        b_A = tl.where(o_i >= j, b_A * scale, 0.)
        tl.store(A + o_A + j, b_A, mask=m_A)
        p_k += K
        p_gk += K

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BC"],
)
@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A,
    A2,
    T: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NK: tl.constexpr
):
    i_t, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_t * BT + i_c * BC >= T:
        return
    n_bh = tl.num_programs(2)
    b_A = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(0, NK):
        p_A = tl.make_block_ptr(A + (i_bh + i_k*n_bh) * T * BC, (T, BC), (BC, 1), (i_t * BT + i_c * BC, 0), (BC, BC), (1, 0))
        b_A += tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    p_A2 = tl.make_block_ptr(A2 + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT + i_c * BC, i_c * BC), (BC, BC), (1, 0))
    tl.store(p_A2, b_A.to(A2.dtype.element_ty), boundary_check=(0, 1))

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["BK", "BV", "BT"],
)
@triton.jit
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
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
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_g = tl.load(p_g, boundary_check=(0, 1), padding_option="zero")
        b_qg = (b_q * tl.exp(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_A = tl.make_block_ptr(A + i_bh * T * BT, (T, BT), (BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.load(p_A, boundary_check=(0, 1), padding_option="zero")
    b_A = tl.where(m_s, b_A, 0.).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v, allow_tf32=False)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

def _require_fp32_contiguous(reference, *tensors):
    for tensor in (reference, *tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("all tensor arguments must be tensors")
        if tensor.layout != torch.strided or tensor.dtype != torch.float32 or not tensor.is_cuda:
            raise ValueError("chunk_gla_fwd requires strided CUDA FP32 tensors")
        if tensor.is_neg() or tensor.is_conj():
            raise ValueError("chunk_gla_fwd requires resolved tensor view flags")
        if tensor.device != reference.device or not tensor.is_contiguous():
            raise ValueError("chunk_gla_fwd requires contiguous tensors on one CUDA device")


def _require_query(q, BT):
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a tensor")
    if q.ndim != 4 or q.shape[-1] <= 0:
        raise ValueError("q must have shape [B,H,T,K] with K > 0")
    # The output kernel forms a BT by BT tile; Triton limits tiles to 2**20 elements.
    if type(BT) is not int or BT not in (16, 32, 64, 128, 256, 512, 1024):
        raise ValueError("BT must be a power of two from 16 through 1024")


def _require_scale(scale):
    if type(scale) not in (float, int):
        raise TypeError("scale must be a builtin float or int")
    try:
        result = float(scale)
    except OverflowError as error:
        raise ValueError("scale must be representable as finite FP32") from error
    if not math.isfinite(result) or abs(result) > 3.4028234663852886e38:
        raise ValueError("scale must be representable as finite FP32")
    return result


def _launch_bounds(B, H, T, K, V, BT, producer):
    heads = B * H
    NT, NK, NC = triton.cdiv(T, BT), triton.cdiv(K, 128), BT // 16
    sizes = (heads * T * K, heads * T * BT, NK * heads * T * 16,
             heads * T * V, heads * NT * K * V)
    # This includes masked lane addresses and the last increment of every loop.
    bound = max(sizes) + BT * max(K, V, BT, 16) + 256 * max(V, 1) + 256
    if max(B, H, T, K, V, BT, NT, NK, heads) >= 2**31 or bound >= 2**31:
        raise ValueError("source integer arithmetic exceeds signed 32-bit range")
    if producer:
        grids = [(NT, NC * NC, heads), (NT, NC, heads)]
        if K > 256:
            grids.append((NK, NT * NC, heads))
    else:
        BV = min(32, triton.next_power_of_2(max(1, V)))
        grids = [(triton.cdiv(V, BV), NT, heads)]
    if any(x >= 2**31 or y > 65535 or z > 65535 for x, y, z in grids):
        raise ValueError("launch grid exceeds CUDA axis limits")
    return bound


def _require_addresses(bound, *tensors):
    for tensor in tensors:
        pointer = tensor.data_ptr()
        if pointer % 4 or pointer < 0 or pointer + 4 * bound >= 2**64:
            raise ValueError("FP32 pointer alignment or address range is invalid")


def chunk_fwd_intra_gated_gk_fn(q, k, g, scale, BT):
    _require_query(q, BT)
    _require_fp32_contiguous(q, k, g)
    if k.shape != q.shape or g.shape != q.shape:
        raise ValueError("q, k and g must have matching [B,H,T,K] shapes")
    scale = _require_scale(scale)
    B, H, T, K = q.shape
    BC = 16
    NC = triton.cdiv(BT, BC)
    NT = triton.cdiv(T, BT)
    BK = min(64, triton.next_power_of_2(max(16, K)))
    if B * H * T:
        bound = _launch_bounds(B, H, T, K, 0, BT, True)
        _require_addresses(bound, q, k, g)
    A = q.new_zeros((B, H, T, BT), dtype=torch.float32)
    if B * H * T == 0:
        return A
    _require_addresses(bound, A)
    with torch.cuda.device(q.device):
        grid = (NT, NC * NC, B * H)
        chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
            q, k, g, A, T * K, K, scale,
            T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
        )
        grid = (NT, NC, B * H)
        if K <= 256:
            BK = triton.next_power_of_2(K)
            chunk_gla_fwd_A_kernel_intra_sub_intra[grid](
                q, k, g, A, T * K, K, scale,
                T=T, K=K, BT=BT, BC=BC, BK=BK
            )
        else:
            BK = 128
            NK = triton.cdiv(K, BK)
            A_intra = q.new_zeros((NK, B, H, T, BC), dtype=torch.float32)
            _require_addresses(bound, A_intra)
            grid = (NK, NT * NC, B * H)
            chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](
                q, k, g, A_intra, T * K, K, scale,
                T=T, K=K, BT=BT, BC=BC, BK=BK, NC=NC
            )
            grid = (NT, NC, B * H)
            chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](
                A_intra, A, T=T, BT=BT, BC=BC, NK=NK
            )
    return A


def chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, BT, scale):
    _require_query(q, BT)
    _require_fp32_contiguous(q, v, g_cumsum, A, h)
    B, H, T, K = q.shape
    if v.ndim != 4 or tuple(v.shape[:3]) != (B, H, T) or g_cumsum.shape != q.shape:
        raise ValueError("v and gate must match q batch/head/time axes")
    scale = _require_scale(scale)
    V = v.shape[-1]
    NT = triton.cdiv(T, BT)
    if tuple(A.shape) != (B, H, T, BT):
        raise ValueError("A must have shape [B,H,T,BT]")
    if tuple(h.shape) != (B, H, NT * K, V):
        raise ValueError("history must have shape [B,H,ceil(T/BT)*K,V]")
    BK = min(32, triton.next_power_of_2(max(16, K)))
    BV = min(32, triton.next_power_of_2(max(1, V)))
    NV = triton.cdiv(V, BV)
    if B * H * T * V:
        bound = _launch_bounds(B, H, T, K, V, BT, False)
        _require_addresses(bound, q, v, g_cumsum, A, h)
    o = torch.empty_like(v)
    if B * H * T * V == 0:
        return o
    _require_addresses(bound, o)
    with torch.cuda.device(q.device):
        chunk_gla_fwd_kernel_o[(NV, NT, B * H)](
            q, v, g_cumsum, h, o, A,
            T * K, K, T * V, V, NT * K * V, V, scale,
            T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
        )
    return o

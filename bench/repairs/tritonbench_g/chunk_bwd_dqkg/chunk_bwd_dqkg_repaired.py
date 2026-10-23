
import math
import struct
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_simple_gla_bwd_kernel_dqkg(
    q,
    k,
    v,
    h,
    g,
    do,
    dh,
    dq,
    dk,
    dg,
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
    BV: tl.constexpr,
    NT: tl.constexpr
):
    # Kernel implementation
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_g = tl.make_block_ptr(g + i_bh * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    b_g = tl.load(p_g, boundary_check=(0,), padding_option="zero")
    last_idx = min(i_t * BT + BT, T) - 1
    b_g_last = tl.load(g + i_bh * T + last_idx)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    b_dg_last = tl.zeros([1,], dtype=tl.float32)
    b_dg = tl.zeros([BT,], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * (K * s_h_t), (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * (K * s_h_t), (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")

        b_dg_last += (tl.sum(b_h * b_dh))
        b_ds += tl.dot(b_do, tl.trans(b_v))
        b_dq += tl.dot(b_do, b_h.to(b_do.dtype))
        b_dk += tl.dot(b_v, b_dh.to(b_v.dtype))

    p_q = tl.make_block_ptr(q + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_dg_last *= tl.exp(b_g_last)
    b_dq = b_dq * tl.exp(b_g)[:, None] * scale
    b_dk = b_dk * tl.exp(-b_g + b_g_last)[:, None]
    b_dg_last += tl.sum(b_dk * b_k)
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale * tl.exp(b_g[:, None] - b_g[None, :]), 0)
    b_ds = b_ds.to(b_k.dtype)
    # [BT, BK]
    b_dq += tl.dot(b_ds, b_k)
    b_dk += tl.dot(tl.trans(b_ds), b_q)
    b_dg += tl.sum(b_q * b_dq - b_k * b_dk, axis=1)
    # (SY 09/21) revcumsum in a separate kernel due to strange triton compiler issue
    # b_dg = tl.dot(tl.where(o_i[:, None] <= o_i[None, :], 1., 0.), b_dg, allow_tf32=False) + b_dg_last)
    b_dg = tl.where(o_i < min(BT, T-i_t*BT) - 1, b_dg, b_dg + b_dg_last)
    p_dq = tl.make_block_ptr(dq + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_k_h, (T, K), (s_k_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dg = tl.make_block_ptr(dg + (i_k*n_bh + i_bh) * T, (T,), (1,), (i_t * BT,), (BT,), (0,))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0,))

def _canonical_strides(shape):
    result = []
    stride = 1
    for size in reversed(shape):
        result.append(stride)
        stride *= max(size, 1)
    return tuple(reversed(result))


def _validate_inputs(do, q, k, v, g, h, dh, scale):
    """The selected dense FP32 ABI, including one history matrix per chunk."""
    inputs = (q, k, v, g, do, h, dh)
    if not all(isinstance(x, torch.Tensor) for x in inputs):
        raise ValueError("all inputs must be tensors")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have four dimensions")
    B, H, T, K = k.shape
    V = v.shape[-1]
    if K <= 0 or V <= 0:
        raise ValueError("key and value dimensions must be positive")
    NT = (T + 63) // 64
    shapes = ((B, H, T, K), (B, H, T, K), (B, H, T, V),
              (B, H, T), (B, H, T, V), (B, H, NT * K, V),
              (B, H, NT * K, V))
    for tensor, shape in zip(inputs, shapes):
        if tuple(tensor.shape) != shape or tuple(tensor.stride()) != _canonical_strides(shape):
            raise ValueError("input shape or canonical dense strides do not match the ABI")
        if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
            raise ValueError("the selected storage dtype is FP32")
        if tensor.device != q.device or not tensor.is_cuda:
            raise ValueError("inputs must share one CUDA device")
        if not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous")
        if tensor.is_conj() or tensor.is_neg():
            raise ValueError("lazy tensor flags are outside this ABI")
        if tensor.numel() and tensor.data_ptr() % 4:
            raise ValueError("FP32 input pointers must be naturally aligned")
    if not isinstance(scale, (float, int)) or isinstance(scale, bool):
        raise ValueError("scale must be a finite scalar")
    try:
        scale32 = struct.unpack("f", struct.pack("f", float(scale)))[0]
    except (OverflowError, ValueError):
        raise ValueError("scale must be representable as finite FP32") from None
    if not math.isfinite(scale32):
        raise ValueError("scale must be representable as finite FP32")
    if B and H and T:
        spans = [(x.data_ptr(), x.data_ptr() + 4 * x.numel()) for x in inputs]
        for i, (left, right) in enumerate(spans):
            for other_left, other_right in spans[i + 1:]:
                if left < other_right and other_left < right:
                    raise ValueError("input data spans must be pairwise disjoint")
        BK = max(16, min(triton.next_power_of_2(K), 64))
        BV = max(16, min(triton.next_power_of_2(V), 64))
        NK = triton.cdiv(K, BK)
        largest = max(B * H * T * K, B * H * T * V,
                      B * H * NT * K * V, NK * B * H * T)
        margin = (64 + BK + BV + 1) * max(T, K, V, NT * K, K * V, 1)
        if largest + margin >= 2**31:
            raise ValueError("formed source indices exceed the signed 32-bit envelope")
        if NK >= 2**31 or NT > 65535 or B * H > 65535:
            raise ValueError("grid exceeds CUDA dimension limits")
        if any(x.data_ptr() + 4 * (largest + margin) >= 2**64 for x in inputs):
            raise ValueError("formed byte pointers would wrap")
    return B, H, T, K, V, scale32


def chunk_bwd_dqkg_fn(do, q, k, v, g, h, dh, scale):
    B, H, T, K, V, scale = _validate_inputs(do, q, k, v, g, h, dh, scale)
    BT = 64
    BK = max(16, min(triton.next_power_of_2(K), 64))
    BV = max(16, min(triton.next_power_of_2(V), 64))
    NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
    grid = (NK, NT, B * H)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dg = torch.empty(NK, B, H, T, dtype=torch.float32, device=g.device).fill_(-1e9)
    if not B or not H or not T:
        return dq, dk, dg
    largest = max(B * H * T * K, B * H * T * V,
                  B * H * NT * K * V, NK * B * H * T)
    margin = (64 + BK + BV + 1) * max(T, K, V, NT * K, K * V, 1)
    if any(x.data_ptr() + 4 * (largest + margin) >= 2**64 for x in (dq, dk, dg)):
        raise ValueError("formed output byte pointers would wrap")
    chunk_simple_gla_bwd_kernel_dqkg[grid](
        q, k, v, h, g, do, dh, dq, dk, dg,
        q.stride(1), q.stride(2),
        v.stride(1), v.stride(2),
        dh.stride(1), dh.stride(2),
        scale,
        T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT
    )
    return dq, dk, dg

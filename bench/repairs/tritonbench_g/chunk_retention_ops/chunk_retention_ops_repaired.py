
import torch
import triton
import triton.language as tl
from typing import Tuple
import math

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_fwd_kernel_h(
    k, v, h, h0, ht, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    # Triton kernel code for forward pass of chunk retention with initial and final state handling
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        if i_t == NT - 1 and (T % BT) != 0:
            d_b = tl.math.exp2((T % BT) * b_b)
            d_i = tl.math.exp2(((T % BT) - o_i - 1) * b_b)
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=False)
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_fwd_kernel_o(
    q, k, v, h, o, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    scale, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    # Triton kernel code for forward pass of chunk retention with output scaling
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_i = tl.math.exp2((o_i + 1) * b_b)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_o += tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s *= d_s
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_bwd_kernel_dh(
    q, do, dh, d_ht, d_h0, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    scale, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
    USE_FINAL_GRAD: tl.constexpr, STORE_INITIAL_GRAD: tl.constexpr
):
    # Reverse carry includes the optional terminal loss and yields the initial-state adjoint.
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b)
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRAD:
        p_dht = tl.make_block_ptr(d_ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dht, boundary_check=(0, 1), padding_option="zero")
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        actual_length = tl.minimum(BT, T - i_t * BT)
        d_b = tl.math.exp2(actual_length * b_b)
        b_dh = d_b * b_dh + tl.dot(b_q, (b_do * d_i[:, None]).to(b_q.dtype), allow_tf32=False)
    if STORE_INITIAL_GRAD:
        p_dh0 = tl.make_block_ptr(d_h0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4)
    ],
    key=["BT", "BK", "BV"],
)
@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q, k, v, h, do, dh, dq, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h, s_vo_t, s_vo_d, s_h_h, s_h_t,
    scale, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr
):
    # Triton kernel code for backward pass of chunk retention, computing gradients for q, k, v
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    n_bh = tl.num_programs(2).to(tl.int64)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))
    o_i = tl.arange(0, BT)
    actual_length = tl.minimum(BT, T - i_t * BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((actual_length - o_i - 1) * b_b)
    d_q = (d_q * scale).to(d_q.dtype)
    m_s = o_i[:, None] >= o_i[None, :]
    d_s = tl.where(m_s, tl.math.exp2((o_i[:, None] - o_i[None, :]) * b_b), 0) * scale
    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * tl.trans(d_s)
    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (V, K), (1, s_h_t), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh)*s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    b_ds = (b_ds * d_s).to(b_q.dtype)
    b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))
    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))



def _validate_tensor(x, shape, device, name, canonical=True, contiguous=True):
    if not isinstance(x, torch.Tensor) or tuple(x.shape) != tuple(shape):
        raise ValueError(name + " has the wrong shape")
    if (x.dtype != torch.float32 or x.device != device or not x.is_cuda
            or x.layout != torch.strided or x.is_conj() or x.is_neg()
            or (contiguous and not x.is_contiguous())):
        raise ValueError(name + " must be resolved strided FP32 storage on the input CUDA device")
    expected, running = [], 1
    for size in reversed(shape):
        expected.append(running)
        running *= size
    if canonical and tuple(x.stride()) != tuple(reversed(expected)):
        raise ValueError(name + " must use canonical dense strides")
    strides = tuple(x.stride())
    base, capacity = x.untyped_storage().data_ptr(), x.untyped_storage().nbytes()
    offset = x.storage_offset()
    last = offset + sum((size - 1) * stride for size, stride in zip(shape, strides))
    if (any(stride < 0 for stride in strides) or offset < 0
            or base < 0 or base % 4 != 0 or capacity < 0
            or base + capacity > 2 ** 64 or x.data_ptr() != base + 4 * offset
            or not 0 <= last < capacity // 4):
        raise ValueError(name + " must have aligned, bounded FP32 backing storage")


def _overlaps(a, b):
    ap, bp = a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()
    return ap < bp + b.untyped_storage().nbytes() and bp < ap + a.untyped_storage().nbytes()


def _validate_kv(k, v, BT=64):
    if not isinstance(k, torch.Tensor) or k.dim() != 4:
        raise ValueError("k must have four dimensions")
    B, H, T, K = k.shape
    if not isinstance(v, torch.Tensor) or v.dim() != 4:
        raise ValueError("v must have four dimensions")
    V = v.shape[-1]
    if type(BT) is not int or BT != 64 or min(B, H, T, K, V) <= 0 or B * H > 65535:
        raise ValueError("positive dimensions, BT=64 and B*H<=65535 are required")
    NT, NK = triton.cdiv(T, 64), triton.cdiv(K, 64)
    if max(NT, triton.cdiv(V, 64)) > 65535:
        raise ValueError("source launch grid.y must not exceed 65535")
    if max(NT * 64 * K, NT * 64 * V, NT * K * V, triton.cdiv(K, 64) * 64, triton.cdiv(V, 64) * 64) >= 2 ** 31:
        raise ValueError("source-local signed32 offsets exceed the admitted range")
    if max(B * H * NT * K * V, NK * B * H * T * V) * 4 >= 2 ** 63:
        raise ValueError("source signed64 byte capacities exceed the admitted range")
    _validate_tensor(k, (B, H, T, K), k.device, "k")
    _validate_tensor(v, (B, H, T, V), k.device, "v")
    return B, H, T, K, V


def _validate_primal(q, k, v, initial_state=None):
    B, H, T, K, V = _validate_kv(k, v)
    _validate_tensor(q, k.shape, k.device, "q")
    if initial_state is not None:
        _validate_tensor(initial_state, (B, H, K, V), k.device, "initial_state")
    return B, H, T, K, V


def _validate_scale(scale):
    if type(scale) is not float or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("scale must be a finite FP32-representable Python float")


def chunk_fwd_h_fn(k, v, BT, initial_state, output_final_state):
    B, H, T, K, V = _validate_kv(k, v, BT)
    if initial_state is not None:
        _validate_tensor(initial_state, (B, H, K, V), k.device, "initial_state")
    if type(output_final_state) is not bool:
        raise ValueError("output_final_state must be bool")
    with torch.cuda.device(k.device):
        B, H, T, K, V = *k.shape, v.shape[-1]
        final_state = None
        if output_final_state:
            final_state = k.new_empty(B, H, K, V, dtype=torch.float32)
            _validate_tensor(final_state, (B, H, K, V), k.device, "final_state")
        BK, BV = max(16, min(64, triton.next_power_of_2(K))), max(16, min(64, triton.next_power_of_2(V)))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        h = k.new_empty(B, H, NT * K, V)
        _validate_tensor(h, (B, H, NT * K, V), k.device, "h")
        grid = (NK, NV, B * H)
        chunk_retention_fwd_kernel_h[grid](
            k, v, h, initial_state, final_state,
            k.stride(1), k.stride(2), k.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=output_final_state
        )
        return h, final_state


def chunk_fwd_o_fn(h, q, k, v, BT, scale):
    B, H, T, K, V = _validate_primal(q, k, v)
    if type(BT) is not int or BT != 64:
        raise ValueError("BT must be 64")
    _validate_scale(scale)
    _validate_tensor(h, (B, H, triton.cdiv(T, BT) * K, V), k.device, "h")
    with torch.cuda.device(k.device):
        B, H, T, K, V = *k.shape, v.shape[-1]
        o = torch.empty_like(v)
        _validate_tensor(o, v.shape, k.device, "o")
        BK = max(16, min(triton.next_power_of_2(K), 64))
        BV = max(16, min(triton.next_power_of_2(V), 64))
        NV = triton.cdiv(V, BV)
        NT = triton.cdiv(T, BT)
        grid = (NV, NT, B * H)
        chunk_retention_fwd_kernel_o[grid](
            q, k, v, h, o,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            scale,
            H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV
        )
        return o


def chunk_bwd_dh_fn(do, q, k, v, BT, scale, d_ht=None, d_h0=None):
    B, H, T, K, V = _validate_primal(q, k, v)
    if type(BT) is not int or BT != 64:
        raise ValueError("BT must be 64")
    _validate_scale(scale)
    _validate_tensor(do, v.shape, k.device, "do", canonical=False)
    if d_ht is not None:
        _validate_tensor(d_ht, (B, H, K, V), k.device, "d_ht", canonical=False)
    if d_h0 is not None:
        _validate_tensor(d_h0, (B, H, K, V), k.device, "d_h0")
        reads = [q, k, v, do] + ([] if d_ht is None else [d_ht])
        if any(_overlaps(d_h0, x) for x in reads):
            raise ValueError("d_h0 backing storage must be disjoint from the inputs")
    with torch.cuda.device(k.device):
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT = 64
        BK = max(16, min(triton.next_power_of_2(K), 64))
        BV = max(16, min(triton.next_power_of_2(V), 64))
        NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
        dh = k.new_empty(B, H, NT * K, V)
        _validate_tensor(dh, (B, H, NT * K, V), k.device, "dh")
        grid = (NK, NV, B * H)
        chunk_retention_bwd_kernel_dh[grid](
            q, do, dh, d_ht, d_h0,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            dh.stride(1), dh.stride(2),
            scale,
            H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
            USE_FINAL_GRAD=d_ht is not None, STORE_INITIAL_GRAD=d_h0 is not None
        )
        return dh


def chunk_bwd_dqkv_fn(do, q, k, v, h, dh, scale):
    B, H, T, K, V = _validate_primal(q, k, v)
    _validate_scale(scale)
    _validate_tensor(do, v.shape, k.device, "do", canonical=False)
    _validate_tensor(h, (B, H, triton.cdiv(T, 64) * K, V), k.device, "h")
    _validate_tensor(dh, h.shape, k.device, "dh")
    with torch.cuda.device(k.device):
        B, H, T, K, V = *k.shape, v.shape[-1]
        BT = 64
        BK = max(16, min(triton.next_power_of_2(K), 64))
        BV = max(16, min(triton.next_power_of_2(V), 64))
        NT, NK = triton.cdiv(T, BT), triton.cdiv(K, BK)
        grid = (NK, NT, B * H)
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = v.new_empty(NK, *v.shape)
        _validate_tensor(dq, q.shape, k.device, "dq")
        _validate_tensor(dk, k.shape, k.device, "dk")
        _validate_tensor(dv, (NK, *v.shape), k.device, "dv_partial")
        chunk_retention_bwd_kernel_dqkv[grid](
            q, k, v, h, do, dh, dq, dk, dv,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            h.stride(1), h.stride(2),
            scale,
            H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT
        )
        dv = dv.sum(0)
        _validate_tensor(dv, v.shape, k.device, "dv")
        return dq, dk, dv


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, initial_state, output_final_state, scale, checkpoint_level):
        BT = 64
        h, final_state = chunk_fwd_h_fn(k, v, BT, initial_state, output_final_state)
        o = chunk_fwd_o_fn(h, q, k, v, BT, scale)
        if checkpoint_level == 1:
            h = None
        ctx.save_for_backward(q, k, v, h, initial_state)
        ctx.BT, ctx.scale = BT, scale
        ctx.output_final_state = output_final_state
        return o.to(q.dtype), final_state

    @staticmethod
    def backward(ctx, do, d_ht=None):
        if torch.is_grad_enabled():
            raise ValueError("only first-order backward is supported")
        BT, scale = ctx.BT, ctx.scale
        q, k, v, h, initial_state = ctx.saved_tensors
        B, H, T, K, V = _validate_primal(q, k, v, initial_state)
        if do is not None:
            _validate_tensor(do, v.shape, q.device, "do", canonical=False, contiguous=False)
        if d_ht is not None:
            if not ctx.output_final_state:
                raise ValueError("a final-state cotangent requires a returned final state")
            _validate_tensor(d_ht, (B, H, K, V), q.device, "d_ht", canonical=False, contiguous=False)
        with torch.cuda.device(q.device):
            # Autograd may supply expanded cotangents (for example loss.sum()).
            # Their logical values are preserved while satisfying the existing kernel ABI.
            do = torch.zeros_like(v) if do is None else do.contiguous()
            d_ht = None if d_ht is None else d_ht.contiguous()
            if h is None:
                h, _ = chunk_fwd_h_fn(k, v, BT, initial_state, False)
            d_h0 = torch.empty_like(initial_state) if initial_state is not None and ctx.needs_input_grad[3] else None
            dh = chunk_bwd_dh_fn(do, q, k, v, BT, scale, d_ht, d_h0)
            dq, dk, dv = chunk_bwd_dqkv_fn(do, q, k, v, h, dh, scale)
            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), d_h0, None, None, None


def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    scale: float = None,
    checkpoint_level: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    _validate_primal(q, k, v, initial_state)
    if type(checkpoint_level) is not int or checkpoint_level not in (0, 1):
        raise ValueError("checkpoint_level must be 0 or 1")
    if type(output_final_state) is not bool:
        raise ValueError("output_final_state must be bool")
    if scale is None:
        scale = q.size(-1) ** -0.5
    _validate_scale(scale)
    with torch.cuda.device(q.device):
        return ChunkRetentionFunction.apply(
            q, k, v, initial_state, output_final_state, scale, checkpoint_level)

from typing import Tuple
import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd
import functools


def contiguous(fn):
    def copy_argument(value):
        if isinstance(value, torch.Tensor):
            if type(value) is not torch.Tensor:
                raise TypeError("checked contiguous inputs must be ordinary tensors")
            return value.contiguous()
        return value
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx,
                  *(copy_argument(i) for i in args),
                  **{k: copy_argument(v) for k, v in kwargs.items()})
    return wrapper

@triton.jit
def chunk_retention_fwd_kernel_h(
    k,
    v,
    h,
    initial_state,  # initial state of the chunk [B, H, D_head_K, D_head_V]
    final_state,  # final state of the chunk [B, H, D_head_K, D_head_V]
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    # [BK, BV]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(initial_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    for i_t in range(NT):
        p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BK, BV]
        if i_t == NT - 1 and (T % BT) != 0:
            d_b = tl.math.exp2((T % BT) * b_b)
            d_i = tl.math.exp2(((T % BT) - o_i - 1) * b_b)
        b_h = d_b * b_h + tl.dot(b_k, (b_v * d_i[:, None]).to(b_k.dtype), allow_tf32=False)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(final_state + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_retention_fwd_kernel_o(
    q,
    k,
    v,
    h,
    o,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
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
        # [BT, BK]
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        # [BK, BT]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BK, BV]
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_o += tl.dot((b_q * d_i[:, None]).to(b_q.dtype), b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    b_s *= d_s
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_retention_bwd_kernel_dh(
    q,
    do,
    dh,
    d_ht,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr,
    USE_FINAL_GRAD: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_b, d_i = tl.math.exp2(BT * b_b), tl.math.exp2((o_i + 1) * b_b)
    # [BK, BV]
    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRAD:
        p_dht = tl.make_block_ptr(d_ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dht, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_o = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        # Save the adjoint after this chunk before propagating to its start.
        tl.store(p_h, b_dh.to(p_h.dtype.element_ty), boundary_check=(0, 1))
        # [BT, BV]
        b_o = tl.load(p_o, boundary_check=(0, 1), padding_option="zero")
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        if i_t == NT - 1 and (T % BT) != 0:
            d_b = tl.math.exp2((T % BT) * b_b)
        else:
            d_b = tl.math.exp2(BT * b_b)
        b_dh = d_b * b_dh + tl.dot(b_q, (b_o * d_i[:, None] * scale).to(b_o.dtype), allow_tf32=False)


@triton.jit
def chunk_retention_bwd_kernel_dqkv(
    q,
    k,
    v,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    s_h_h,
    s_h_t,
    scale,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    i_h = i_bh % H
    n_bh = tl.num_programs(2).to(tl.int64)
    b_b = tl.math.log2(1 - tl.math.exp2(-5 - i_h * 1.0))

    o_i = tl.arange(0, BT)
    d_q, d_k = tl.math.exp2((o_i + 1) * b_b), tl.math.exp2((BT - o_i - 1) * b_b)
    if i_t == NT - 1 and (T % BT) != 0:
        d_k = tl.math.exp2(((T % BT) - o_i - 1) * b_b)
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
        p_dv = tl.make_block_ptr(dv + (i_k * n_bh + i_bh) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        # [BT, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        # [BV, BK]
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        # [BK, BV]
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")

        # [BT, BT]
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        # [BT, BK]
        b_dq += tl.dot(b_do, b_h, allow_tf32=False)
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        # [BT, BV]
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) * d_k[:, None] + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    # [BT, BT]
    b_ds = (b_ds * d_s).to(b_q.dtype)
    # [BT, BK]
    b_dq = b_dq * d_q[:, None] + tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk = b_dk * d_k[:, None] + tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))

    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def _dense_fp32(name, tensor, shape=None, device=None):
    if type(tensor) is not torch.Tensor:
        raise TypeError(name + " must be a tensor")
    if tensor.dtype != torch.float32 or not tensor.is_cuda:
        raise ValueError(name + " must be CUDA FP32")
    if tensor.layout != torch.strided or not tensor.is_contiguous():
        raise ValueError(name + " must have a contiguous dense layout")
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError(name + " must have resolved lazy bits")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(name + " has an incompatible shape")
    if device is not None and tensor.device != device:
        raise ValueError(name + " is on a different device")


def _plan(B, H, T, K, V):
    if min(B, H, T) < 0 or K <= 0 or V <= 0:
        raise ValueError("invalid dimensions")
    BT = 64
    BK = max(16, min(64, triton.next_power_of_2(K)))
    BV = max(16, min(64, triton.next_power_of_2(V)))
    NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
    BH, PT, PK, PV = B * H, NT * BT, NK * BK, NV * BV
    if BH and max(BH, NT, NV) > 65535:
        raise OverflowError("CUDA grid y/z axis exceeds 65535")
    metadata = (B, H, T, K, V, BT, BK, BV, NT, NK, NV, BH, PT, PK, PV,
                T * K, T * V, NT * K, NT * K * V)
    if any(x > 2**31 - 1 for x in metadata):
        raise OverflowError("source metadata or padded coordinates exceed signed32")
    # Extents cover all block-pointer lanes, including inactive padding.
    q_extent = (BH - 1) * T * K + (PT - 1) * K + PK if BH and NT else 0
    v_extent = (BH - 1) * T * V + (PT - 1) * V + PV if BH and NT else 0
    h_extent = (BH - 1) * NT * K * V + (NT - 1) * K * V + (PK - 1) * V + PV if BH and NT else 0
    state_extent = (BH - 1) * K * V + (PK - 1) * V + PV if BH else 0
    dv_extent = (NK * BH - 1) * T * V + (PT - 1) * V + PV if BH and NT else 0
    extents = (q_extent, v_extent, h_extent, state_extent, dv_extent)
    if any(x < 0 or 4 * x > 2**63 - 1 for x in extents):
        raise OverflowError("formed byte displacement exceeds signed64")
    return BT, BK, BV, NT, NK, NV, extents


def _addresses(*items):
    for tensor, extent in items:
        if tensor is None:
            continue
        base = tensor.data_ptr()
        if tensor.element_size() != 4 or tensor.storage_offset() < 0 or 4 * (tensor.storage_offset() + tensor.numel()) > tensor.untyped_storage().nbytes():
            raise OverflowError("FP32 tensor extent is not backed by its actual storage")
        if base < 0 or base % 4 or base + 4 * max(extent, tensor.numel()) >= 2**64:
            raise OverflowError("formed FP32 pointer exceeds aligned unsigned64")


class ChunkRetentionFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(device_type='cuda')
    @contiguous
    def forward(ctx, q, k, v, initial_state, output_final_state):
        _dense_fp32("q", q)
        if q.ndim != 4:
            raise ValueError("q must have rank four")
        B, H, T, K = q.shape
        _dense_fp32("k", k, q.shape, q.device)
        _dense_fp32("v", v, device=q.device)
        if v.ndim != 4 or tuple(v.shape[:3]) != (B, H, T):
            raise ValueError("v must share the batch, head and time dimensions")
        V = v.shape[-1]
        if type(output_final_state) is not bool:
            raise TypeError("output_final_state must be boolean")
        if initial_state is not None:
            _dense_fp32("initial_state", initial_state, (B, H, K, V), q.device)
        BT, BK, BV, NT, NK, NV, extents = _plan(B, H, T, K, V)
        q_extent, v_extent, h_extent, state_extent, _ = extents
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K ** -0.5
        final_state = q.new_empty(B, H, K, V, dtype=torch.float32, requires_grad=False) if output_final_state else None
        h = q.new_empty(B, H, NT * K, V)
        o = torch.empty_like(v)
        _addresses((q, q_extent), (k, q_extent), (v, v_extent), (h, h_extent),
                   (o, v_extent), (initial_state, state_extent), (final_state, state_extent))
        # Canonical strides also cover dense singleton tensors with unusual reported strides.
        qk_strides, vo_strides, h_strides = (T * K, K, 1), (T * V, V, 1), (NT * K * V, V)
        if B * H:
            with torch.cuda.device(q.device):
                chunk_retention_fwd_kernel_h[(NK, NV, B * H)](
                    k, v, h, initial_state, final_state,
                    *qk_strides, *vo_strides, *h_strides,
                    H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                    USE_INITIAL_STATE=initial_state is not None,
                    STORE_FINAL_STATE=output_final_state,
                    num_warps=num_warps, num_stages=num_stages)
                if NT:
                    chunk_retention_fwd_kernel_o[(NV, NT, B * H)](
                        q, k, v, h, o, *qk_strides, *vo_strides, *h_strides, scale,
                        H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
                        num_warps=num_warps, num_stages=num_stages)
        ctx.save_for_backward(q, k, v, h)
        ctx.output_final_state = output_final_state
        return o.to(q.dtype), final_state

    @staticmethod
    @custom_bwd(device_type='cuda')
    @contiguous
    def backward(ctx, do, d_ht=None):
        q, k, v, h = ctx.saved_tensors
        B, H, T, K = q.shape
        V = v.shape[-1]
        if do is None:
            do = torch.zeros_like(v)
        _dense_fp32("do", do, v.shape, q.device)
        if d_ht is not None:
            if not ctx.output_final_state:
                raise ValueError("final-state gradient requires a returned final state")
            _dense_fp32("d_ht", d_ht, (B, H, K, V), q.device)
        BT, BK, BV, NT, NK, NV, extents = _plan(B, H, T, K, V)
        q_extent, v_extent, h_extent, state_extent, dv_extent = extents
        num_stages = 1
        num_warps = 4 if BK == 64 else 2
        scale = K ** -0.5
        dh = q.new_empty(B, H, NT * K, V)
        dq, dk = torch.empty_like(q), torch.empty_like(k)
        dv = v.new_empty(NK, *v.shape)
        _addresses((q, q_extent), (k, q_extent), (v, v_extent), (h, h_extent),
                   (do, v_extent), (d_ht, state_extent), (dh, h_extent),
                   (dq, q_extent), (dk, q_extent), (dv, dv_extent))
        qk_strides, vo_strides, h_strides = (T * K, K, 1), (T * V, V, 1), (NT * K * V, V)
        if B * H and NT:
            with torch.cuda.device(q.device):
                chunk_retention_bwd_kernel_dh[(NK, NV, B * H)](
                    q, do, dh, d_ht, *qk_strides, *vo_strides, *h_strides, scale,
                    H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                    USE_FINAL_GRAD=d_ht is not None,
                    num_warps=num_warps, num_stages=num_stages)
                chunk_retention_bwd_kernel_dqkv[(NK, NT, B * H)](
                    q, k, v, h, do, dh, dq, dk, dv,
                    *qk_strides, *vo_strides, *h_strides, scale,
                    H=H, T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                    num_warps=num_warps, num_stages=num_stages)
        dv = dv.sum(0)
        _addresses((dv, v_extent))
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, None


def chunk_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        if type(initial_state) is not torch.Tensor:
            raise TypeError("initial_state must be an ordinary tensor")
        initial_state = initial_state.detach()
    return ChunkRetentionFunction.apply(q, k, v, initial_state, output_final_state)


import torch
import math
import triton
import triton.language as tl
from typing import Optional, Tuple

@triton.jit
def chunk_linear_attn_fwd_kernel_h(
    k, v, h, h0, ht,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_h_h, s_h_t,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NT: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

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
        b_h += tl.dot(b_k, b_v, allow_tf32=False)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_linear_attn_fwd_kernel_o(
    q, k, v, h, o,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_h_h, s_h_t, scale,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    qk_head = i_bh * s_qk_h
    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q + qk_head, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k + qk_head, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)
    b_s = tl.where(m_s, b_s, 0)

    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

    b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_linear_attn_bwd_kernel_dh(
    q, do, dh, dht, dh0,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_h_h, s_h_t, scale,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr,
    USE_FINAL_GRADIENT: tl.constexpr, STORE_INITIAL_GRADIENT: tl.constexpr
):
    i_k, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    b_dh = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRADIENT:
        p_dht = tl.make_block_ptr(dht + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_dh = tl.load(p_dht, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    for i_t in range(NT - 1, -1, -1):
        p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h + i_t * K * V, (K, V), (s_h_t, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        tl.store(p_dh, b_dh.to(p_dh.dtype.element_ty), boundary_check=(0, 1))
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
        b_q = (b_q * scale).to(b_q.dtype)
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_dh += tl.dot(b_q, b_do.to(b_q.dtype), allow_tf32=False)

    if STORE_INITIAL_GRADIENT:
        p_dh0 = tl.make_block_ptr(dh0 + i_bh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_dh0, b_dh.to(p_dh0.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def chunk_linear_attn_bwd_kernel_dqkv(
    q, k, v, h, do, dh, dq, dk, dv,
    s_qk_h, s_qk_t, s_qk_d,
    s_vo_h, s_vo_t, s_vo_d,
    s_h_h, s_h_t, scale,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NT: tl.constexpr
):
    i_k, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n_bh = tl.num_programs(2)
    o_i = tl.arange(0, BT)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
    b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
    b_s = tl.where(o_i[:, None] <= o_i[None, :], b_s, 0)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_ds = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_h = tl.make_block_ptr(h + i_bh * s_h_h, (V, NT * K), (1, s_h_t), (i_v * BV, i_t * K + i_k * BK), (BV, BK), (0, 1))
        p_do = tl.make_block_ptr(do + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
        p_dh = tl.make_block_ptr(dh + i_bh * s_h_h, (NT * K, V), (s_h_t, 1), (i_t * K + i_k * BK, i_v * BV), (BK, BV), (1, 0))
        p_dv = tl.make_block_ptr(dv + (i_k*n_bh+i_bh)*s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero")
        b_h = tl.load(p_h, boundary_check=(0, 1), padding_option="zero")
        b_dh = tl.load(p_dh, boundary_check=(0, 1), padding_option="zero")

        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)
        b_dq += tl.dot(b_do, b_h, allow_tf32=False) * scale
        b_dk += tl.dot(b_v, tl.trans(b_dh), allow_tf32=False)
        b_dv = tl.dot(b_k, b_dh, allow_tf32=False) + tl.dot(b_s.to(b_q.dtype), b_do, allow_tf32=False)
        tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    b_ds = tl.where(o_i[:, None] >= o_i[None, :], b_ds * scale, 0).to(b_q.dtype)
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.trans(tl.dot(b_q, b_ds, allow_tf32=False))

    p_dq = tl.make_block_ptr(dq + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    p_dk = tl.make_block_ptr(dk + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def _formed_bound(B, H, T, K, V):
    NT = triton.cdiv(T, 64)
    NK = triton.cdiv(K, min(64, triton.next_power_of_2(K)))
    return max(B*H*T*K, B*H*T*V, B*H*NT*K*V, NK*B*H*T*V) + 256*max(K,V)


def _check_pointers(bound, *tensors):
    for tensor in tensors:
        if tensor is not None:
            pointer = tensor.data_ptr()
            if pointer < 0 or pointer % 4 or pointer + 4*bound >= 2**63:
                raise ValueError("FP32 pointer alignment or finite address range is invalid")


def _check_cotangent(tensor, reference, shape):
    if tensor is not None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("cotangents must be tensors")
        if (tensor.layout != torch.strided or tensor.dtype != torch.float32 or not tensor.is_cuda
                or tensor.device != reference.device or tensor.is_neg() or tensor.is_conj()
                or tuple(tensor.shape) != tuple(shape)):
            raise ValueError("cotangent must have the matching shape, dtype, and device")


class ChunkLinearAttentionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale, initial_state, output_final_state):
        with torch.cuda.device(q.device):
            B, H, T, K, V = *q.shape, v.shape[-1]
            BT = 64
            BK, BV = min(64, triton.next_power_of_2(K)), min(64, triton.next_power_of_2(V))
            NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            ctx.scale = scale
            ctx.has_initial_state = initial_state is not None
            ctx.output_final_state = output_final_state
            ctx.bound = _formed_bound(B, H, T, K, V)
            ctx.set_materialize_grads(False)

            final_state = None
            if output_final_state:
                final_state = q.new_empty(B, H, K, V, dtype=torch.float32, requires_grad=False)

            h = q.new_empty(B, H, NT * K, V)
            _check_pointers(ctx.bound, q, k, v, initial_state, h, final_state)
            grid = (NK, NV, B * H)
            chunk_linear_attn_fwd_kernel_h[grid](
                k, v, h, initial_state, final_state,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2),
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                USE_INITIAL_STATE=initial_state is not None,
                STORE_FINAL_STATE=output_final_state,
                num_warps=num_warps,
                num_stages=num_stages
            )
            grid = (NV, NT, B * H)
            o = torch.empty_like(v)
            _check_pointers(ctx.bound, o)
            chunk_linear_attn_fwd_kernel_o[grid](
                q, k, v, h, o,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                h.stride(1), h.stride(2),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV,
                num_warps=num_warps,
                num_stages=num_stages
            )
            ctx.save_for_backward(q, k, v, h)
            return o.to(q.dtype), final_state
    @staticmethod
    def backward(ctx, do, dht=None):
        q, k, v, h = ctx.saved_tensors
        with torch.cuda.device(q.device):

            B, H, T, K, V = *q.shape, v.shape[-1]
            BT = 64
            BK, BV = min(64, triton.next_power_of_2(K)), min(32 if q.dtype == torch.float32 else 64, triton.next_power_of_2(V))
            NT, NK, NV = triton.cdiv(T, BT), triton.cdiv(K, BK), triton.cdiv(V, BV)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            scale = ctx.scale
            _check_cotangent(do, q, v.shape)
            _check_cotangent(dht, q, (B,H,K,V))
            if dht is not None and not ctx.output_final_state:
                raise ValueError("a terminal cotangent requires a returned final state")
            _check_pointers(ctx.bound, do, dht)
            # Sum losses produce zero-stride expanded cotangents; the kernels use
            # the forward tensors' dense strides. Materialize only these API inputs.
            do = torch.zeros_like(v) if do is None else do.contiguous()
            dht = None if dht is None else dht.contiguous()
            dh0 = q.new_empty(B, H, K, V) if ctx.has_initial_state else None

            dh = q.new_empty(B, H, NT * K, V)
            _check_pointers(ctx.bound, q, k, v, h, do, dht, dh0, dh)
            grid = (NK, NV, B * H)
            chunk_linear_attn_bwd_kernel_dh[grid](
                q, do, dh, dht, dh0,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                dh.stride(1), dh.stride(2),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                USE_FINAL_GRADIENT=dht is not None,
                STORE_INITIAL_GRADIENT=ctx.has_initial_state,
                num_warps=num_warps,
                num_stages=num_stages
            )

            grid = (NK, NT, B * H)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = v.new_empty(NK, *v.shape)
            num_stages = 1
            num_warps = 4 if BK == 64 else 2
            _check_pointers(ctx.bound, dq, dk, dv)
            chunk_linear_attn_bwd_kernel_dqkv[grid](
                q, k, v, h, do, dh, dq, dk, dv,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                dh.stride(1), dh.stride(2),
                scale,
                T=T, K=K, V=V, BT=BT, BK=BK, BV=BV, NT=NT,
                num_warps=num_warps,
                num_stages=num_stages
            )
            dv = dv.sum(0)
            _check_pointers(ctx.bound, dv)
            return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None, dh0, None

def chunk_linear_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    scale: Optional[float] = None, initial_state: torch.Tensor = None,
    output_final_state: bool = False, normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    tensors = [q, k, v] + ([] if initial_state is None else [initial_state])
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("Q, K, V and optional initial state must be tensors")
    if type(normalize) is not bool or type(output_final_state) is not bool:
        raise ValueError("state and compatibility flags must be bool")
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("expected Q/K[B,H,T,K] and V[B,H,T,V]")
    B, H, T, K, V = *q.shape, v.shape[-1]
    if min(B, H, T) <= 0 or min(K, V) < 16:
        raise ValueError("positive B/H/T and K,V >= 16 required")
    tensors = [q, k, v] + ([] if initial_state is None else [initial_state])
    if any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda or t.device != q.device
           or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("ordinary materialized same-device FP32 tensors required")
    if q.stride() != (H*T*K, T*K, K, 1) or k.stride() != q.stride() or v.stride() != (H*T*V, T*V, V, 1):
        raise ValueError("Q/K/V must use canonical dense storage")
    if initial_state is not None and (initial_state.shape != (B,H,K,V) or not initial_state.is_contiguous()):
        raise ValueError("initial_state must be contiguous [B,H,K,V]")
    if scale is None:
        scale = K ** -0.5
    if type(scale) not in (float, int):
        raise ValueError("scale must be a finite FP32-range Python scalar")
    try:
        scale = float(scale)
    except OverflowError as error:
        raise ValueError("scale must be a finite FP32-range Python scalar") from error
    if not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("scale must be a finite FP32-range Python scalar")
    scale = float(scale)
    # normalize is the original compatibility no-op; both flags intentionally
    # select the same unnormalized causal attention. No normalized mode exists.
    NT, NK = triton.cdiv(T, 64), triton.cdiv(K, min(64, triton.next_power_of_2(K)))
    NV_forward = triton.cdiv(V, min(64, triton.next_power_of_2(V)))
    NV_backward = triton.cdiv(V, min(32, triton.next_power_of_2(V)))
    # All four launch tuples use these axes. Check before allocation/dispatch.
    if max(B*H, NT, NK, NV_forward, NV_backward) > 65535:
        raise ValueError("kernel launch axes exceed the selected 65535 limit")
    if max(B*H*T*K, B*H*T*V, B*H*NT*K*V, NK*B*H*T*V) + 256*max(K,V) >= 2**31:
        raise ValueError("formed tensor offsets exceed the selected signed-int32 contract")
    _check_pointers(_formed_bound(B, H, T, K, V), *tensors)
    o, final_state = ChunkLinearAttentionFunction.apply(q, k, v, scale, initial_state, output_final_state)
    return o, final_state




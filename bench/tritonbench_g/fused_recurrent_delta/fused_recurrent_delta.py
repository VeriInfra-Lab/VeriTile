
import torch
import triton
import triton.language as tl
from typing import Tuple
import math
import struct

@triton.jit
def fused_recurrent_fwd_kernel(
    q, k, v, beta, o, h0, ht, s_qk_h, s_vo_h, scale, B, H, T, K: tl.constexpr, V: tl.constexpr, 
    BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, 
    IS_HEADWISE_BETA: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + i_bh * T
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _v_minus = tl.sum(h * b_k[None, :], axis=1)
        b_v -= _v_minus
        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        tl.store(p_v, b_v.to(p_v.dtype.element_ty), mask=mask_bv)
        b_v *= b_beta
        h += b_k[None, :] * b_v[:, None]
        _o = h * b_q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += K
        p_k += K
        p_o += V
        p_v += V
        p_beta += V if IS_HEADWISE_BETA else 1

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, h.to(p_ht.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_bwd_kernel(
    q, k, v, beta, dht, dh0, do, dq, dk, dv, dbeta, h0, s_qk_h, s_vo_h, NK, scale, B, H, T, 
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, 
    IS_HEADWISE_BETA: tl.constexpr, USE_DH0: tl.constexpr, USE_DHT: tl.constexpr
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    mask_bk = i_k * BK + tl.arange(0, BK) < K
    mask_bv = i_v * BV + tl.arange(0, BV) < V

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    else:
        p_beta = beta + i_bh * T + T - 1

    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * K
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    if IS_HEADWISE_BETA:
        p_dbeta = dbeta + (i_bh + i_k * B * H + i_v * B * H * NK) * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * V
    else:
        p_dbeta = dbeta + (i_bh + i_v * B * H) * T + T - 1
    d_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_DHT:
        p_ht = dht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        d_h += tl.load(p_ht, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    for _ in range(T):
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        d_h += b_q[:, None] * b_do[None, :]
        d_k = tl.sum(d_h * (b_v * b_beta)[None, :], axis=1)
        d_v = tl.sum(d_h * b_k[:, None], axis=0)

        d_beta = d_v * b_v if IS_HEADWISE_BETA else tl.sum(d_v * b_v)
        d_v = d_v * b_beta

        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)
        if IS_HEADWISE_BETA:
            tl.store(p_dbeta, d_beta.to(p_dbeta.dtype.element_ty), mask=mask_bv)
        else:
            tl.store(p_dbeta, d_beta.to(p_dbeta.dtype.element_ty))

        d_h -= b_k[:, None] * d_v[None, :]

        p_do -= V
        p_q -= K
        p_k -= K
        p_v -= V
        p_dk -= K
        p_dv -= V
        p_dbeta -= V if IS_HEADWISE_BETA else 1
        p_beta -= V if IS_HEADWISE_BETA else 1

    if USE_DH0:
        p_dh0 = dh0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        tl.store(p_dh0, d_h.to(p_dh0.dtype.element_ty), mask=mask_bk[:, None] & mask_bv[None, :])

    tl.debug_barrier()

    h = tl.zeros([BK, BV], dtype=tl.float32)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    if IS_HEADWISE_BETA:
        p_beta = beta + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    else:
        p_beta = beta + i_bh * T
    p_do = do + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[:, None]) * V + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        d_k = tl.load(p_dk, mask=mask_bk, other=0).to(tl.float32)
        d_v = tl.load(p_dv, mask=mask_bv, other=0).to(tl.float32)
        d_k -= tl.sum(d_v[None, :] * h, axis=1)
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)

        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        if IS_HEADWISE_BETA:
            b_beta = tl.load(p_beta, mask=mask_bv, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta

        h += b_k[:, None] * b_v[None, :]
        _d_q = h * b_do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += K
        p_do += V
        p_v += V
        p_dk += K
        p_dv += V
        p_dq += K
        p_beta += V if IS_HEADWISE_BETA else 1

def _tensor(tensor, shape, device):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a tensor")
    if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
        raise TypeError("only strided FP32 tensors are supported")
    if tensor.device.type != "cuda" or tensor.device != device:
        raise ValueError("all tensors must use the same CUDA device")
    if tuple(tensor.shape) != tuple(shape) or not tensor.is_contiguous():
        raise ValueError("tensor must have the expected dense shape")
    step, expected = 1, []
    for dimension in reversed(shape):
        expected.append(step)
        step *= max(dimension, 1)
    if tensor.stride() != tuple(reversed(expected)):
        raise ValueError("source strides must be canonical even on singleton axes")
    if tensor.is_neg() or tensor.is_conj():
        raise ValueError("lazy view flags are unsupported")


def _shape(q, k, v, output_final_state):
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise TypeError("q must be a rank-four tensor")
    if not isinstance(v, torch.Tensor) or v.ndim != 4:
        raise TypeError("v must be a rank-four tensor")
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be a bool")
    B, H, T, K = q.shape
    V = v.shape[-1]
    if min(B, H, T, K, V) <= 0 or B * H > 65535:
        raise ValueError("dimensions must be positive and the head grid must fit")
    _tensor(q, (B, H, T, K), q.device)
    _tensor(k, (B, H, T, K), q.device)
    _tensor(v, (B, H, T, V), q.device)
    return B, H, T, K, V


def _spans(shape, vector):
    B, H, T, K, V = shape
    BK = triton.next_power_of_2(K)
    FBV, BBV = min(triton.next_power_of_2(V), 8), min(triton.next_power_of_2(V), 32)
    FNV, BNV = triton.cdiv(V, FBV), triton.cdiv(V, BBV)
    if FNV >= 1 << 31 or BNV >= 1 << 31 or BK * BBV > 1048576:
        raise ValueError("grid or block dimensions exceed the native domain")
    sizes = {}
    for name in ("q", "k", "v", "beta", "h0", "o", "ht", "do", "dht", "dq", "dk", "dv", "dbeta", "dh0", "retq", "retk", "retv", "retbeta"):
        rows = K if name in ("h0", "ht", "dht", "dh0") else T
        cols = K if name in ("q", "k", "dq", "dk", "retq", "retk") else (V if vector else 1) if name in ("beta", "dbeta", "retbeta") else V
        planes = BNV * B * H if name in ("dq", "dk", "dbeta") else B * H
        span = planes * rows * cols + rows * cols + cols + BK * V + BBV + BK
        if span >= 1 << 31:
            raise ValueError("formed source indices must fit signed32")
        sizes[name] = span
    return sizes


def _windows(tensors, spans, shape):
    lower = max(shape[3], shape[4], 1)
    windows = []
    for name, tensor in tensors:
        base, size = tensor.data_ptr(), 4 * tensor.numel()
        if base % 4 or base < 4 * lower or base + 4 * spans[name] + 4 > 1 << 64:
            raise ValueError("formed native pointers must remain aligned and in range")
        windows.append((base, base + size))
    for i, (lo, hi) in enumerate(windows):
        if any(not (hi <= other_lo or other_hi <= lo) for other_lo, other_hi in windows[i + 1:]):
            raise ValueError("source regions must occupy disjoint byte windows")


def _forward_inputs(q, k, v, beta, scale, initial_state, output_final_state):
    shape = _shape(q, k, v, output_final_state)
    B, H, T, K, V = shape
    vector = beta is not None and isinstance(beta, torch.Tensor) and beta.ndim == 4
    tensors = [("q", q), ("k", k), ("v", v)]
    if beta is not None:
        _tensor(beta, (B, H, T, V) if vector else (B, H, T), q.device)
        tensors.append(("beta", beta))
    if initial_state is not None:
        _tensor(initial_state, (B, H, K, V), q.device)
        tensors.append(("h0", initial_state))
    if scale is not None:
        if type(scale) is not float or scale <= 0:
            raise ValueError("scale must be a positive Python float")
        try:
            launch = struct.unpack("f", struct.pack("f", scale))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError("scale must have a finite FP32 launch value") from error
        if not math.isfinite(launch):
            raise ValueError("scale must have a finite FP32 launch value")
    spans = _spans(shape, vector)
    _windows(tensors, spans, shape)
    return shape, spans, tensors


class FusedRecurrentFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, beta, scale=None, initial_state=None, output_final_state=False):
        shape, spans, tensors = _forward_inputs(q, k, v, beta, scale, initial_state, output_final_state)
        B, H, T, K, V = shape

        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1
        assert NK == 1, "NK > 1 is not supported yet"
        o = q.new_empty(NK, B, H, T, V)

        if output_final_state:
            final_state = q.new_empty(B, H, K, V, dtype=torch.float32)
        else:
            final_state = None

        _windows(tensors + [("o", o)] + ([("ht", final_state)] if final_state is not None else []), spans, shape)
        grid = (NV, NK, B * H)
        with torch.cuda.device(q.device):
            fused_recurrent_fwd_kernel[grid](
                q, k, v, beta, o, initial_state, final_state,
                q.stride(1),
                v.stride(1),
                scale,
                B=B, H=H, T=T, K=K, V=V,
                BK=BK, BV=BV,
                USE_INITIAL_STATE=initial_state is not None,
                STORE_FINAL_STATE=final_state is not None,
                IS_HEADWISE_BETA=beta.ndim == v.ndim,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        o = o.squeeze(0)
        ctx.save_for_backward(q, k, v, beta, initial_state)
        ctx.scale = scale
        ctx._used = False
        ctx._output_final = output_final_state
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        if getattr(ctx, "_used", False) or torch.is_grad_enabled():
            raise RuntimeError("only one ordinary first-order backward is supported")
        q, k, v, beta, initial_state = ctx.saved_tensors
        shape, spans, tensors = _forward_inputs(q, k, v, beta, ctx.scale, initial_state, ctx._output_final)
        B, H, T, K, V = shape
        _tensor(do, (B, H, T, V), q.device)
        tensors.append(("do", do))
        if dht is not None:
            if not ctx._output_final:
                raise ValueError("a terminal cotangent requires a returned final state")
            _tensor(dht, (B, H, K, V), q.device)
            tensors.append(("dht", dht))
        _windows(tensors, spans, shape)
        ctx._used = True
        B, H, T, K, V = *q.shape, v.shape[-1]
        scale = ctx.scale
        BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        assert NK == 1, "NK > 1 is not supported yet"
        num_stages = 1
        num_warps = 2

        beta_vector = beta.ndim == v.ndim

        dq = q.new_empty(NV, B, H, T, K)
        dk = q.new_empty(NV, B, H, T, K)
        dv = q.new_empty(NK, B, H, T, V)
        if beta_vector:
            dbeta = q.new_zeros(NV, NK, B, H, T, V)
        else:
            dbeta = q.new_empty(NV, B, H, T)
        grid = (NV, NK, B * H)

        if initial_state is not None and initial_state.requires_grad:
            dh0 = torch.empty_like(initial_state, dtype=torch.float32)
        else:
            dh0 = None

        scratch = [("dq", dq), ("dk", dk), ("dv", dv), ("dbeta", dbeta)]
        if dh0 is not None:
            scratch.append(("dh0", dh0))
        _windows(tensors + scratch, spans, shape)
        with torch.cuda.device(q.device):
            fused_recurrent_bwd_kernel[grid](
                q, k, v, beta, dht, dh0, do, dq, dk, dv, dbeta, initial_state,
                q.stride(1),
                v.stride(1),
                NK, scale,
                B=B, H=H, T=T, K=K, V=V,
                BK=BK, BV=BV,
                USE_INITIAL_STATE=initial_state is not None,
                USE_DH0=dh0 is not None,
                USE_DHT=dht is not None,
                IS_HEADWISE_BETA=beta_vector,
                num_warps=num_warps,
                num_stages=num_stages
            )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        dbeta = dbeta.sum((0, 1)) if beta_vector else dbeta.sum(0)
        return dq.to(q), dk.to(k), dv.to(v), dbeta.to(beta), None, dh0, None

def fused_recurrent_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    _forward_inputs(q, k, v, beta, scale, initial_state, output_final_state)
    if scale is None:
        scale = q.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"
    if beta is None:
        beta = torch.ones_like(q[..., 0])
    o, final_state = FusedRecurrentFunction.apply(q, k, v, beta, scale, initial_state, output_final_state)
    return o, final_state

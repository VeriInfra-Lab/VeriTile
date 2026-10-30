
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_retention_fwd_kernel(
    q, k, v, o, initial_state, final_state, 
    s_qk_h, s_qk_t, s_qk_d, 
    s_vo_h, s_vo_t, s_vo_d, 
    B, H, T, scale, 
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, 
    USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr
):
    # Kernel logic
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = (1 - tl.math.exp2(-5 - i_h * 1.0))

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)

    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    h = tl.zeros([BV, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale

        h = b_b * h + _k[None, :] * _v[:, None]
        _o = h * _q[None, :]
        _o = tl.sum(_o, axis=1)
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)

        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)

@triton.jit
def fused_recurrent_retention_bwd_kernel(
    q, k, v, do, dq, dk, dv, initial_state, d_final_state,
    s_do_b, s_do_h, s_do_t, s_do_d,
    s_df_b, s_df_h, s_df_k, s_df_v,
    s_qk_h, s_qk_t, s_qk_d, 
    s_vo_h, s_vo_t, s_vo_d, 
    B, H, T, scale, 
    BK: tl.constexpr, BV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, 
    USE_INITIAL_STATE: tl.constexpr, USE_FINAL_GRAD: tl.constexpr
):
    # Kernel logic
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H
    b_b = 1 - tl.math.exp2(-5 - i_h * 1.0)

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_do = do + (i_bh // H) * s_do_b + i_h * s_do_h + (i_v * BV + tl.arange(0, BV)) * s_do_d

    p_dq = dq + (i_bh + i_v * B * H) * s_qk_h + i_k * BK + tl.arange(0, BK)
    mask_bk = i_k * BK + tl.arange(0, BK) < DK
    mask_bv = i_v * BV + tl.arange(0, BV) < DV

    h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        mask_kv = mask_bk[:, None] & mask_bv[None, :]
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * \
            DV + (i_v * BV + tl.arange(0, BV)[None, :])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    for i in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)

        h = b_b * h + _k[:, None] * _v[None, :]
        _d_q = h * _do[None, :]
        d_q = tl.sum(_d_q, axis=1) * scale
        tl.store(p_dq, d_q.to(p_dq.dtype.element_ty), mask=mask_bk)

        p_k += DK
        p_do += s_do_t
        p_v += DV
        p_dq += DK

    tl.debug_barrier()

    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK) + (T - 1) * DK
    p_do = do + (i_bh // H) * s_do_b + i_h * s_do_h + (i_v * BV + tl.arange(0, BV)) * s_do_d + (T - 1) * s_do_t
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV) + (T - 1) * DV
    p_dk = dk + (i_bh + i_v * B * H) * s_qk_h + i_k * \
        BK + tl.arange(0, BK) + (T - 1) * DK
    p_dv = dv + (i_bh + i_k * B * H) * s_vo_h + i_v * \
        BV + tl.arange(0, BV) + (T - 1) * DV
    d_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_FINAL_GRAD:
        p_df = d_final_state + (i_bh // H) * s_df_b + i_h * s_df_h + \
            (i_k * BK + tl.arange(0, BK)[:, None]) * s_df_k + \
            (i_v * BV + tl.arange(0, BV)[None, :]) * s_df_v
        d_h = tl.load(p_df, mask=mask_bk[:, None] & mask_bv[None, :], other=0).to(tl.float32)

    for _ in range(T):
        _do = tl.load(p_do, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        d_h += _q[:, None] * _do[None, :]
        d_k = tl.sum(d_h * _v[None, :], axis=1)
        d_v = tl.sum(d_h * _k[:, None], axis=0)

        d_h *= b_b
        tl.store(p_dk, d_k.to(p_dk.dtype.element_ty), mask=mask_bk)
        tl.store(p_dv, d_v.to(p_dv.dtype.element_ty), mask=mask_bv)

        p_do -= s_do_t
        p_q -= DK
        p_k -= DK
        p_v -= DV
        p_dk -= DK
        p_dv -= DV

class _Radius:
    """Magnitude bound, plus safety of every signed32 subexpression."""
    def __init__(self, radius, safe=True):
        self.radius = radius
        self.safe = safe and radius < 1 << 31
    def __add__(self, other):
        if not isinstance(other, _Radius):
            other = _Radius(abs(other))
        return _Radius(self.radius + other.radius, self.safe and other.safe)
    def __sub__(self, other):
        return self + other
    def __mul__(self, other):
        if not isinstance(other, _Radius):
            other = _Radius(abs(other))
        return _Radius(self.radius * other.radius, self.safe and other.safe)


def _tensor_view(tensor, shape, device, dense):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a tensor")
    if tensor.dtype != torch.float32 or tensor.layout != torch.strided or tensor.ndim != 4:
        raise TypeError("expected a rank-four strided FP32 tensor")
    if tensor.device.type != "cuda" or tensor.device != device:
        raise ValueError("all buffers must use the same CUDA device")
    if tuple(tensor.shape) != tuple(shape) or (dense and not tensor.is_contiguous()):
        raise ValueError("unexpected tensor shape or dense layout")
    canonical = (max(shape[1], 1) * max(shape[2], 1) * max(shape[3], 1), max(shape[2], 1) * max(shape[3], 1), max(shape[3], 1), 1)
    if dense and tensor.stride() != canonical:
        raise ValueError("source strides must be canonical even on singleton axes")
    if tensor.is_neg() or tensor.is_conj() or any(stride < 0 for stride in tensor.stride()):
        raise ValueError("unsupported tensor view metadata")
    offset = tensor.storage_offset()
    capacity = tensor.untyped_storage().nbytes() // 4
    strides = tensor.stride()
    if tensor.numel() and offset + sum((dim - 1) * stride for dim, stride in zip(shape, strides)) >= capacity:
        raise ValueError("strided reads exceed storage")
    base = tensor.data_ptr() - 4 * offset
    if base < 0 or base % 4 or base + 4 * capacity > 1 << 64:
        raise ValueError("unaligned or overflowing storage window")
    return (offset, capacity, strides, base)


def _address_radius(shape, views, backward, seed, extra):
    B, H, T, DK, DV = shape
    BK, BV = min(DK, 32), min(DV, 32)
    NK, NV = triton.cdiv(DK, BK), triton.cdiv(DV, BV)
    args = (B, H, T, DK, DV, BK, BV, NK, NV, B * H, NV * NK, NV * NK * B * H)
    if any(value >= 1 << 31 for value in args) or NK > 65535 or B * H > 65535:
        raise ValueError("native scalar or launch grid overflow")
    q, k, v, do, initial, final = views
    if any(value >= 1 << 31 for value in q[2][1:] + v[2][1:]):
        raise ValueError("source strides must fit signed32")
    if backward and any(value >= 1 << 31 for value in do[2] + (final[2] if extra else ())):
        raise ValueError("cotangent strides must fit signed32")
    n = _Radius
    iv, ik, h, lk, lv, batch, head, time = map(n, (NV - 1, NK - 1, B * H - 1, BK - 1, BV - 1, B - 1, H - 1, T))
    key, val = ik * n(BK) + lk, iv * n(BV) + lv
    roots = list(map(n, (B, H, T, DK, DV, BK, BV, NK, NV))) + [n(B) * n(H), iv, ik, h, lk, lv, head, key, val]
    roots += list(map(n, q[2][1:] + v[2][1:])) + [h * n(q[2][1]), h * n(v[2][1])]
    partial_o = (h + ik * n(B) * n(H)) * n(v[2][1])
    partial_q = (h + iv * n(B) * n(H)) * n(q[2][1])
    state_roots = [h * n(DK) * n(DV), key * n(DV)]
    if not backward:
        roots += [partial_o] + (state_roots if seed or extra else [])
    else:
        roots += [partial_o, partial_q, batch] + list(map(n, do[2]))
        roots += [batch * n(do[2][0]), head * n(do[2][1]), val * n(do[2][3]),
                  (n(T) - n(1)) * n(DK), (n(T) - n(1)) * n(DV), (n(T) - n(1)) * n(do[2][2])]
        if seed:
            roots += state_roots
        if extra:
            roots += list(map(n, final[2])) + [batch * n(final[2][0]), head * n(final[2][1]),
                key * n(final[2][2]), val * n(final[2][3])]
    if not all(root.safe for root in roots):
        raise ValueError("a source integer subexpression may overflow signed32")
    qp = lambda t: n(q[0]) + h * n(q[2][1]) + key + t * n(DK)
    kp = lambda t: n(k[0]) + h * n(q[2][1]) + key + t * n(DK)
    vp = lambda t: n(v[0]) + h * n(v[2][1]) + val + t * n(DV)
    dp = lambda t: n(do[0]) + batch * n(do[2][0]) + head * n(do[2][1]) + val * n(do[2][3]) + t * n(do[2][2])
    ov = lambda t: partial_o + val + t * n(DV)
    ok = lambda t: partial_q + key + t * n(DK)
    state = h * n(DK) * n(DV) + key * n(DV) + val
    initial_pointers = [n(initial[0]) + state] if seed else []
    if not backward:
        formed = [qp(time), kp(time), vp(time), ov(time)] + initial_pointers + ([state] if extra else [])
    else:
        rev = n(T) - n(1) - time
        formed = [qp(n(0)), kp(time), vp(time), dp(time), ok(time), qp(rev), kp(rev), vp(rev), dp(rev), ok(rev), ov(rev)] + initial_pointers
        if extra:
            formed += [n(final[0]) + batch * n(final[2][0]) + head * n(final[2][1]) + key * n(final[2][2]) + val * n(final[2][3])]
    radius = max(pointer.radius for pointer in formed)
    if 4 * radius >= 1 << 63:
        raise ValueError("formed pointer displacements exceed signed64")
    return radius


def _pointer_windows(tensors, radius):
    for tensor in tensors:
        base = tensor.data_ptr() - 4 * tensor.storage_offset()
        if base < 4 * radius or base + 4 * radius + 4 > 1 << 64 or base % 4:
            raise ValueError("formed native pointers overflow their absolute domain")


def _check_forward(q, k, v, initial_state, output_final_state):
    if not isinstance(q, torch.Tensor) or not isinstance(v, torch.Tensor) or q.ndim != 4 or v.ndim != 4:
        raise TypeError("q and v must be rank-four tensors")
    if type(output_final_state) is not bool:
        raise TypeError("output_final_state must be a Python bool")
    B, H, T, DK = q.shape
    DV = v.shape[-1]
    if B <= 0 or H <= 0 or T <= 0 or DK <= 0 or DV <= 0 or min(DK, 32) not in (1, 2, 4, 8, 16, 32) or min(DV, 32) not in (1, 2, 4, 8, 16, 32):
        raise ValueError("positive dimensions and supported source arange widths are required")
    shape = (B, H, T, DK, DV)
    qv = _tensor_view(q, (B, H, T, DK), q.device, True)
    kv = _tensor_view(k, (B, H, T, DK), q.device, True)
    vv = _tensor_view(v, (B, H, T, DV), q.device, True)
    seed = _tensor_view(initial_state, (B, H, DK, DV), q.device, True) if initial_state is not None else qv
    views = (qv, kv, vv, qv, seed, qv)
    radius = _address_radius(shape, views, False, initial_state is not None, output_final_state)
    tensors = [q, k, v] + ([initial_state] if initial_state is not None else [])
    _pointer_windows(tensors, radius)
    return shape, views, radius, tensors


def _check_backward(q, k, v, initial_state, output_final_state, do, d_final_state):
    shape, views, _, tensors = _check_forward(q, k, v, initial_state, output_final_state)
    B, H, T, DK, DV = shape
    if torch.is_grad_enabled():
        raise RuntimeError("higher-order backward is unsupported")
    if d_final_state is not None and not output_final_state:
        raise ValueError("a final-state cotangent requires a returned final state")
    do_view = _tensor_view(do, (B, H, T, DV), q.device, False)
    final_view = _tensor_view(d_final_state, (B, H, DK, DV), q.device, False) if d_final_state is not None else views[0]
    views = (views[0], views[1], views[2], do_view, views[4], final_view)
    radius = _address_radius(shape, views, True, initial_state is not None, d_final_state is not None)
    tensors += [do] + ([d_final_state] if d_final_state is not None else [])
    _pointer_windows(tensors, radius)
    return radius, tensors


class FusedRecurrentRetentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, initial_state=None, output_final_state=False):
        _, _, radius, tensors = _check_forward(q, k, v, initial_state, output_final_state)
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]

        scale = d_head_qk ** -0.5
        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        o = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)

        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
        else:
            final_state = None

        _pointer_windows(tensors + [o] + ([final_state] if final_state is not None else []), radius)
        grid = (NV, NK, batch_size * n_heads)
        with torch.cuda.device(q.device):
            fused_recurrent_retention_fwd_kernel[grid](
                q, k, v, o, initial_state, final_state,
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                batch_size, n_heads, seq_len, scale,
                DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
                num_warps=num_warps,
                num_stages=num_stages,
                USE_INITIAL_STATE=initial_state is not None,
                STORE_FINAL_STATE=final_state is not None
            )

        o = o.sum(0)
        ctx.save_for_backward(q, k, v, initial_state)
        ctx._output_final = output_final_state
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        q, k, v, initial_state = ctx.saved_tensors
        radius, tensors = _check_backward(q, k, v, initial_state, ctx._output_final, do, d_final_state)
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        d_head_v = v.shape[-1]
        scale = d_head_qk ** -0.5

        BK, BV = min(d_head_qk, 32), min(d_head_v, 32)
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        num_stages = 1
        num_warps = 1

        dq = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dk = q.new_empty(NV, batch_size, n_heads, seq_len, d_head_qk)
        dv = q.new_empty(NK, batch_size, n_heads, seq_len, d_head_v)
        grid = (NV, NK, batch_size * n_heads)

        _pointer_windows(tensors + [dq, dk, dv], radius)
        with torch.cuda.device(q.device):
            fused_recurrent_retention_bwd_kernel[grid](
                q, k, v, do, dq, dk, dv, initial_state, d_final_state,
                *do.stride(),
                *(d_final_state.stride() if d_final_state is not None else (0, 0, 0, 0)),
                q.stride(1), q.stride(2), q.stride(3),
                v.stride(1), v.stride(2), v.stride(3),
                batch_size, n_heads, seq_len, scale,
                DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
                num_warps=num_warps,
                num_stages=num_stages,
                USE_INITIAL_STATE=initial_state is not None,
                USE_FINAL_GRAD=d_final_state is not None
            )
        dq = dq.sum(0)
        dk = dk.sum(0)
        dv = dv.sum(0)
        return dq, dk, dv, None, None

def fused_recurrent_retention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_forward(q, k, v, initial_state, output_final_state)
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentRetentionFunction.apply(q, k, v, initial_state, output_final_state)
    return o, final_state


import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    q, k, v, w, u, o, h0, ht, s_k_h, s_v_h, scale, B: tl.constexpr, H: tl.constexpr, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr, REVERSE: tl.constexpr,
):
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_bh % H

    p_q = q + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_k = k + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_v = v + i_bh * s_v_h + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_o = o + (i_bh + i_k * B * H) * (T * V) + i_v * BV + tl.arange(0, BV) + ((T-1) * V if REVERSE else 0)
    p_w = w + i_bh * s_k_h + i_k * BK + tl.arange(0, BK) + ((T-1) * K if REVERSE else 0)
    p_u = u + i_h * K + tl.arange(0, BK) + i_k * BK

    mask_bk = (i_k * BK + tl.arange(0, BK)) < K
    mask_bv = (i_v * BV + tl.arange(0, BV)) < V
    mask_kv = mask_bv[:, None] & mask_bk[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        b_h += tl.load(p_h0, mask=mask_kv, other=0).to(tl.float32)

    b_u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    for _ in range(0, T):
        b_k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        b_q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        b_w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        b_w = tl.exp(b_w)
        b_kv = b_k[None, :] * b_v[:, None]
        b_o = (b_h + b_kv * b_u[None, :]) * b_q[None, :]
        b_o = tl.sum(b_o, axis=1)
        b_h = b_h * b_w[None, :]
        b_h += b_kv
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += -K if REVERSE else K
        p_k += -K if REVERSE else K
        p_o += -V if REVERSE else V
        p_v += -V if REVERSE else V
        p_w += -K if REVERSE else K

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * K * V + (i_k * BK + tl.arange(0, BK)[None, :]) * V + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_kv)

def _check_tensor(x, name):
    if type(x) is not torch.Tensor:
        raise TypeError(f"{name} must be an ordinary Tensor")
    if x.layout != torch.strided or x.dtype != torch.float32 or x.device.type != "cuda":
        raise ValueError(f"{name} must be a strided FP32 CUDA Tensor")
    if not x.is_contiguous() or x.is_neg() or x.is_conj():
        raise ValueError(f"{name} must be contiguous with resolved logical view flags")


def _index_limit(B, H, T, K, V, BK, BV):
    Z = B * H
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    KP, VP = NK * BK, NV * BV
    # Covers every source setup expression, masked lanes and final advances.
    return (Z + NK + NV + BK + BV + K + V + H +
            Z * T * K + Z * T * V + NK * Z * T * V + Z * K * V +
            KP * V + VP + H * K + KP + T * K + T * V + NK * Z + 32)


def _check_pointer(x, limit, negative=0):
    if x.numel():
        ptr = x.data_ptr()
        if ptr % 4 or ptr < 4 * negative or ptr + 4 * (limit + 1) >= 2**64:
            raise ValueError("active FP32 pointer needs alignment and formed-address room")


def _check_inputs(q, k, v, w, u, initial_state, output_final_state, reverse):
    if type(output_final_state) is not bool or type(reverse) is not bool:
        raise TypeError("output_final_state and reverse must be bool")
    values = [(q, "r"), (k, "k"), (v, "v"), (w, "w"), (u, "u")]
    if initial_state is not None:
        values.append((initial_state, "initial_state"))
    for x, name in values:
        _check_tensor(x, name)
    if q.ndim != 4 or k.shape != q.shape or w.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("expected R/K/W [B,H,T,K] and V [B,H,T,V]")
    B, H, T, K = q.shape
    V = v.shape[-1]
    if min(B, H, K, V) < 1:
        raise ValueError("B, H, K and V must be positive; T may be zero")
    if u.shape != (H, K) or initial_state is not None and initial_state.shape != (B, H, K, V):
        raise ValueError("expected U [H,K] and optional state [B,H,K,V]")
    if any(x.device != q.device for x, _ in values):
        raise ValueError("all inputs must share one CUDA device")
    BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    if NV >= 2**31 or NK > 65535 or B * H > 65535:
        raise ValueError("RWKV launch axes exceed CUDA limits")
    limit = _index_limit(B, H, T, K, V, BK, BV)
    if limit >= 2**31:
        raise ValueError("RWKV setup expressions exceed signed 32-bit range")
    for x, name in values:
        negative = (V if name == "v" else K) if reverse and name in ("r", "k", "v", "w") else 0
        _check_pointer(x, limit, negative)
    return limit


def _check_scale(scale):
    if type(scale) not in (int, float):
        raise TypeError("scale must be a Python int or float")


class FusedRecurrentRWKV6Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, r, k, v, w, u, scale=None, initial_state=None, output_final_state=False, reverse=False):
        q = r
        limit = _check_inputs(q, k, v, w, u, initial_state, output_final_state, reverse)
        _check_scale(scale)
        B, H, T, K, V = *q.shape, v.shape[-1]

        BK, BV = min(triton.next_power_of_2(K), 32), min(triton.next_power_of_2(V), 32)
        NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
        num_stages = 1
        num_warps = 1

        final_state = q.new_empty(B, H, K, V) if output_final_state else None

        o = q.new_empty(NK, B, H, T, V, dtype=torch.float32)
        _check_pointer(o, limit, V if reverse else 0)
        if final_state is not None:
            _check_pointer(final_state, limit)
        grid = (NV, NK, B * H)
        if T or output_final_state:
            # The JIT binds raw Python ints as i32/i64/u64. Do not convert
            # scale to float or restrict the branch that performs no launch.
            if type(scale) is int and not -(2**63) <= scale < 2**64:
                raise ValueError("launched integer scale must fit the Triton i64/u64 ABI")
            with torch.cuda.device(q.device):
                fused_recurrent_rwkv6_fwd_kernel[grid](
                    q, k, v, w, u, o, initial_state, final_state,
                    T * K,
                    T * V,
                    scale,
                    B=B, H=H, T=T, K=K, V=V, BK=BK, BV=BV,
                    USE_INITIAL_STATE=initial_state is not None,
                    STORE_FINAL_STATE=final_state is not None,
                    REVERSE=reverse,
                    num_warps=num_warps,
                    num_stages=num_stages
                )


        o = o.sum(0)
        ctx.save_for_backward(q, k, v, w, u, initial_state)
        ctx.scale = scale
        ctx.reverse = reverse
        return o.to(q.dtype), final_state

def fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: float = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_inputs(r, k, v, w, u, initial_state, output_final_state, False)
    _check_scale(scale)
    if scale == -1:
        scale = r.shape[-1] ** -0.5
    o, final_state = FusedRecurrentRWKV6Function.apply(r, k, v, w, u, scale, initial_state, output_final_state)
    return o, final_state

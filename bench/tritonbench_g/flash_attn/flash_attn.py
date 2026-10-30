import math
import torch
import triton
import triton.language as tl


def _tensor(t, shape=None):
    if type(t) is not torch.Tensor or t.ndim != 4 or t.dtype != torch.float32 or t.layout != torch.strided:
        raise ValueError("expected rank-four FP32 tensor")
    if not t.is_cuda or t.requires_grad or t.is_conj() or t.is_neg():
        raise ValueError("expected materialized CUDA inference tensor")
    if shape is not None and tuple(t.shape) != tuple(shape):
        raise ValueError("Q/K/V shape mismatch")
    if any(n <= 0 or n >= 2**31 for n in t.shape) or any(s <= 0 or s >= 2**31 for s in t.stride()):
        raise ValueError("positive signed32 dimensions and strides required")
    # A sufficient injective positive-stride layout condition, without copying tensors.
    span = 1
    for stride, size in sorted(zip(t.stride(), t.shape)):
        if size > 1:
            if stride < span:
                raise ValueError("overlapping tensor layout")
            span += (size - 1) * stride
    offset = t.storage_offset()
    if offset < 0 or 4 * (offset + span) > t.untyped_storage().nbytes():
        raise ValueError("tensor storage extent")
    b, h, n, d = t.shape
    sb, sh, sn, sd = t.stride()
    # Includes masked rows and the last block-pointer advance, before byte scaling.
    formed = (b - 1) * sb + (h - 1) * sh + (n + 127) * sn + (d - 1) * sd
    if formed >= 2**31:
        raise ValueError("formed tensor offset exceeds signed32")
    pointer = t.data_ptr()
    # Also covers the pinned four-stage profile's masked i64 prefetch pointers.
    pointer_envelope = formed + 512*sn
    if pointer < 0 or pointer % 4 or pointer + 4*(pointer_envelope + 1) >= 2**64:
        raise ValueError("aligned formed tensor byte addresses must fit unsigned64")


def _validate(q, k, v, causal, sm_scale):
    for t in (q, k, v):
        _tensor(t, q.shape)
    if k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V devices differ")
    if len({t.untyped_storage().data_ptr() for t in (q, k, v)}) != 3:
        raise ValueError("independent Q/K/V storage required")
    if q.shape[3] not in (16, 32, 64, 128):
        raise ValueError("unsupported head dimension")
    if type(causal) is not bool or type(sm_scale) not in (int, float):
        raise ValueError("boolean causal and real scalar scale required")
    if not math.isfinite(sm_scale) or abs(sm_scale) > 3.4028234663852886e38:
        raise ValueError("finite FP32 scalar scale required")
    if q.shape[0] * q.shape[1] > 65535 or q.shape[0] * q.shape[1] * q.shape[2] + 127 >= 2**31:
        raise ValueError("launch or scratch signed32 extent")

def flash_attn_triton(q, k, v, causal=True, sm_scale=1):
    _validate(q, k, v, causal, sm_scale)
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    o = torch.empty_like(q)
    _tensor(o, q.shape)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    pointer = L.data_ptr()
    if pointer < 0 or pointer % 4 or pointer + 4*(q.shape[0]*q.shape[1]*q.shape[2] + 128) >= 2**64:
        raise ValueError("aligned formed scratch byte addresses must fit unsigned64")
    num_warps = 4 if Lk <= 64 else 8
    if type(sm_scale) is int and not (-(2**63) <= sm_scale < 2**64):
        raise ValueError("raw integer scale exceeds Triton scalar ABI")
    with torch.cuda.device(q.device):
        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DIM=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=4)

    return o


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L,
    O,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    off_bs = off_bs_head // HEAD
    off_head = off_bs_head % HEAD
    q_base_offset = off_bs * stride_q_bs + off_head * stride_q_head
    k_base_offset = off_bs * stride_k_bs + off_head * stride_k_head
    v_base_offset = off_bs * stride_v_bs + off_head * stride_v_head
    o_base_offset = off_bs * stride_o_bs + off_head * stride_o_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_v_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, SEQLEN) if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk = tl.where(start_n + off_n[None, :] < SEQLEN, qk, float("-inf"))
        qk += tl.dot(q.to(tl.float32), k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16).to(tl.float32), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom), mask=off_m < SEQLEN)
    O_block_ptr = tl.make_block_ptr(
        base=O + o_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16).to(tl.float32), boundary_check=(0,))


# Retained integer head-offset normalization, applied to the repaired pitches.
@triton.jit
def _fwd_kernel_index_commuted(
    Q, K, V, sm_scale,
    L,
    O,
    stride_q_bs, stride_q_head, stride_q_seqlen, stride_q_dim,
    stride_k_bs, stride_k_head, stride_k_seqlen, stride_k_dim,
    stride_v_bs, stride_v_head, stride_v_seqlen, stride_v_dim,
    stride_o_bs, stride_o_head, stride_o_seqlen, stride_o_dim,
    BS, HEAD, SEQLEN,
    BLOCK_M: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bs_head = tl.program_id(1)

    off_bs = off_bs_head // HEAD
    off_head = off_bs_head % HEAD
    q_base_offset = off_bs * stride_q_bs + stride_q_head * off_head
    k_base_offset = off_bs * stride_k_bs + stride_k_head * off_head
    v_base_offset = off_bs * stride_v_bs + stride_v_head * off_head
    o_base_offset = off_bs * stride_o_bs + stride_o_head * off_head
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_q_seqlen, stride_q_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_base_offset,
        shape=(DIM, SEQLEN),
        strides=(stride_k_dim, stride_k_seqlen),
        offsets=(0, 0),
        block_shape=(DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_v_seqlen, stride_v_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_N, DIM),
        order=(1, 0),
    )
    off_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    max = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    denom = tl.zeros([BLOCK_M], dtype=tl.float32)
    out_buffer = tl.zeros([BLOCK_M, DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(tl.float16)
    lo = 0
    hi = tl.minimum((start_m + 1) * BLOCK_M, SEQLEN) if IS_CAUSAL else SEQLEN
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(off_m[:, None] >= (start_n + off_n[None, :]), qk, float("-inf"))
        qk = tl.where(start_n + off_n[None, :] < SEQLEN, qk, float("-inf"))
        qk += tl.dot(q.to(tl.float32), k)

        max_new = tl.maximum(max, tl.max(qk, 1))
        alpha = tl.math.exp2(max - max_new)
        nume = tl.math.exp2(qk - max_new[:, None])
        out_scale = denom * 0 + alpha
        out_buffer *= out_scale[:, None]
        out_buffer += tl.dot(nume.to(tl.float16).to(tl.float32), v)
        denom = denom * alpha + tl.sum(nume, 1)
        max = max_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    out_buffer = out_buffer / denom[:, None]
    l_ptr = L + off_bs_head * SEQLEN + off_m
    tl.store(l_ptr, max + tl.math.log2(denom), mask=off_m < SEQLEN)
    O_block_ptr = tl.make_block_ptr(
        base=O + o_base_offset,
        shape=(SEQLEN, DIM),
        strides=(stride_o_seqlen, stride_o_dim),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, DIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, out_buffer.to(tl.float16).to(tl.float32), boundary_check=(0,))

def flash_attn_triton_index_commuted(q, k, v, causal=True, sm_scale=1):
    _validate(q, k, v, causal, sm_scale)
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    o = torch.empty_like(q)
    _tensor(o, q.shape)

    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    pointer = L.data_ptr()
    if pointer < 0 or pointer % 4 or pointer + 4*(q.shape[0]*q.shape[1]*q.shape[2] + 128) >= 2**64:
        raise ValueError("aligned formed scratch byte addresses must fit unsigned64")
    num_warps = 4 if Lk <= 64 else 8
    if type(sm_scale) is int and not (-(2**63) <= sm_scale < 2**64):
        raise ValueError("raw integer scale exceeds Triton scalar ABI")
    with torch.cuda.device(q.device):
        _fwd_kernel_index_commuted[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, DIM=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=4)

    return o

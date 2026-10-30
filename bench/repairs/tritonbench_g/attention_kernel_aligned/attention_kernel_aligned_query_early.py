"""Checked aligned bias-attention; independent strides and FP16/BF16 storage."""

import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned_fp16(
    Q, K, V, B0, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_b0z, stride_b0h, stride_b0m,
    Z,
    H,
    N_CTX,
    P_SEQ,
    OUT_DTYPE: tl.constexpr,
    BIAS_LAST_SIZE: tl.constexpr,
    B0_NUMEL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_b * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k, out_dtype=OUT_DTYPE)

        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
        qk += (b0 + b1)
        qk = tl.where((start_n + tl.arange(0, BLOCK_N))[None, :] < N_CTX, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0, 1))



@triton.jit
def _fwd_kernel_aligned_bf16(
    Q, K, V, B0, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_b0z, stride_b0h, stride_b0m,
    Z,
    H,
    N_CTX,
    P_SEQ,
    OUT_DTYPE: tl.constexpr,
    BIAS_LAST_SIZE: tl.constexpr,
    B0_NUMEL: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_b * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k, out_dtype=tl.float32).to(OUT_DTYPE)

        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
        qk += (b0 + b1)
        qk = tl.where((start_n + tl.arange(0, BLOCK_N))[None, :] < N_CTX, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0, 1))



def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, o,
                                                 BLOCK_M=64, BLOCK_N=64,
                                                 num_warps=4, num_stages=2):
    """Standard widths 16/32/64/128, warps 4/8, stages 1..4.

    Bias stores 64 block biases followed by 64 within-block biases per row.
    This mutates the caller's nonoverlapping Out and returns None.
    """
    import math
    tensors = (q, k, v, rel_h_w, o)
    if any(type(t) is not torch.Tensor for t in tensors):
        raise TypeError("ordinary tensors required")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in tensors):
        raise TypeError("equal FP16 or BF16 dtypes required")
    if any(t.layout != torch.strided or t.device.type != "cuda" for t in tensors):
        raise ValueError("strided CUDA tensors required")
    if any(t.device != q.device or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("same-device ordinary tensor values required")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape or q.shape != o.shape:
        raise ValueError("Q/K/V/Out must have equal [B,H,N,D] shapes")
    batches, heads, length, dim = q.shape
    widths = (16, 32, 64, 128)
    if any(type(x) is not int for x in (BLOCK_M, BLOCK_N, num_warps, num_stages)):
        raise TypeError("launch parameters must be ordinary integers")
    if dim not in widths or BLOCK_M not in widths or BLOCK_N not in widths or num_warps not in (4, 8) or num_stages not in (1, 2, 3, 4):
        raise ValueError("widths must be 16/32/64/128, warps 4/8, stages 1..4")
    if BLOCK_N == 128 and dim == 128 and (num_stages == 4 or (BLOCK_M == 128 and num_stages == 3)):
        raise ValueError("selected configuration exceeds the SM80 shared-memory limit")
    if rel_h_w.shape != (batches, heads, length, 128) or not rel_h_w.is_contiguous():
        raise ValueError("bias must be contiguous [B,H,N,128]")
    if length > 64 * BLOCK_N:
        raise ValueError("64 block biases require N <= 64 * BLOCK_N")
    if type(sm_scale) not in (int, float) or not math.isfinite(sm_scale) or abs(sm_scale) > 3.4028234663852886e38:
        raise ValueError("sm_scale must be a finite scalar in FP32 range")
    sm_scale = float(sm_scale)
    if batches == 0 or heads == 0 or length == 0:
        return None
    if batches * heads > 65535:
        raise ValueError("launch Y exceeds the selected runtime limit")
    if any(any(s <= 0 for s in t.stride()) for t in tensors):
        raise ValueError("nonempty tensors require positive strides")
    spans = [(t.untyped_storage().data_ptr(),
              t.untyped_storage().data_ptr() + t.untyped_storage().nbytes())
             for t in tensors]
    if any(a[0] < b[1] and b[0] < a[1]
           for i, a in enumerate(spans) for b in spans[i + 1:]):
        raise ValueError("live input and output storage intervals must be disjoint")
    span = 1
    for stride, extent in sorted(zip(o.stride(), o.shape)):
        if extent > 1:
            if stride < span:
                raise ValueError("Out must have nonoverlapping affine coordinates")
            span += (extent - 1) * stride
    limit = 2**31 - 1
    padded = max(triton.cdiv(length, BLOCK_M) * BLOCK_M, triton.cdiv(length, BLOCK_N) * BLOCK_N)
    for t in (q, k, v, o):
        z, h, m, d = t.stride()
        formed = (batches - 1) * z + (heads - 1) * h + padded * m + (dim - 1) * d
        if max(*t.shape, *t.stride(), formed) > limit:
            raise ValueError("source signed-i32 offset envelope exceeded")
    bz, bh, bm, _ = rel_h_w.stride()
    bias_formed = (batches - 1) * bz + (heads - 1) * bh + (padded - 1) * bm + 127
    if max(*rel_h_w.stride(), bias_formed) > limit:
        raise ValueError("source signed-i32 bias envelope exceeded")
    for t, envelope in zip(tensors, (
            (batches - 1) * q.stride(0) + (heads - 1) * q.stride(1) + padded * q.stride(2) + (dim - 1) * q.stride(3),
            (batches - 1) * k.stride(0) + (heads - 1) * k.stride(1) + padded * k.stride(2) + (dim - 1) * k.stride(3),
            (batches - 1) * v.stride(0) + (heads - 1) * v.stride(1) + padded * v.stride(2) + (dim - 1) * v.stride(3),
            bias_formed,
            (batches - 1) * o.stride(0) + (heads - 1) * o.stride(1) + padded * o.stride(2) + (dim - 1) * o.stride(3))):
        if t.data_ptr() % 2 or t.data_ptr() + 2 * envelope + 2 >= 2**64:
            raise ValueError("aligned nonwrapping half-word pointer required")
    kernel = _fwd_kernel_aligned_fp16 if q.dtype == torch.float16 else _fwd_kernel_aligned_bf16
    storage_dtype = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    grid = (triton.cdiv(length, BLOCK_M), batches * heads, 1)
    with torch.cuda.device(q.device):
        kernel[grid](
            q, k, v, rel_h_w, sm_scale, o,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            rel_h_w.stride(0), rel_h_w.stride(1), rel_h_w.stride(2),
            batches, heads, length, 0,
            OUT_DTYPE=storage_dtype, BIAS_LAST_SIZE=64, B0_NUMEL=128,
            BLOCK_DMODEL=dim, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=True)
    return None

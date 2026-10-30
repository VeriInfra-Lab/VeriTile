
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
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
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
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

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M).to(tl.int64)

    b_offset = off_z * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0.0)
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        # , boundary_check=(0, 1), padding_option="zero")
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        # , boundary_check=(1, 0), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k) #, out_dtype=OUT_DTYPE)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias
        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0.0)
        qk += ((b0 + b1) * 1.44269504)
        qk = tl.where(tl.arange(0, BLOCK_N)[None, :] < N_CTX - start_n, qk, -float("inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0,))


_INT32_LIMIT = 1 << 31
_UINT64_LIMIT = 1 << 64


def _launch_envelope(batch, heads, length, dim, bm, bn, strides):
    extent = batch + heads + length + dim + bm + bn + 256 + batch * heads
    for sz, sh, sr, sd in strides:
        extent += batch * sz + heads * sh + (length + bm + bn) * sr + dim * sd + 128
    if extent >= _INT32_LIMIT or batch * heads > 65535:
        raise ValueError("source integer or grid-y envelope exceeded")
    return extent


def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, rel_h_w, sm_scale, o,
                                                 BLOCK_M, BLOCK_N, num_warps, num_stages):
    """FP16/BF16 attention on dense rows with independent batch/head pitches.

    Legal Triton launch plans are a compiler/runtime contract; configuration
    rejection propagates before execution. No input/output copies are made.
    """
    import math
    tensors = (q, k, v, rel_h_w, o)
    if any(type(t) is not torch.Tensor for t in tensors):
        raise TypeError("ordinary torch.Tensor arguments required")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("strided tensors required")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(t.dtype != q.dtype for t in tensors):
        raise TypeError("matching FP16 or BF16 tensors required")
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must use one CUDA device")
    if any(t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("resolved tensors required")
    if q.ndim != 4 or any(t.shape != q.shape for t in (k, v, o)):
        raise ValueError("matching [B,H,N,D] Q/K/V/output required")
    batch, heads, length, dim = q.shape
    if rel_h_w.shape != (batch, heads, length, 128) or not rel_h_w.is_contiguous():
        raise ValueError("contiguous [B,H,N,128] relative bias required")
    if any(type(x) is not int for x in (BLOCK_M, BLOCK_N, num_warps, num_stages)):
        raise TypeError("integer launch parameters required")
    if BLOCK_M not in (16, 32, 64, 128) or BLOCK_N not in (16, 32, 64, 128):
        raise ValueError("query/key blocks must be powers of two from 16 through 128")
    if dim not in (16, 32, 64, 128) or num_warps not in (4, 8) or not 1 <= num_stages <= 4:
        raise ValueError("unsupported tensor-core launch configuration")
    if type(sm_scale) not in (int, float) or not math.isfinite(sm_scale) or abs(sm_scale) > 3.4028234663852886e38:
        raise ValueError("finite FP32-representable scale required")
    if triton.cdiv(length, BLOCK_N) > 128:
        raise ValueError("key-block bias index exceeds the 128-column bias")
    if batch == 0 or heads == 0 or length == 0:
        return None
    for t in (q, k, v, o):
        sz, sh, sr, sd = t.stride()
        if sr != dim or sd != 1 or sh < length * dim or sz < heads * sh:
            raise ValueError("dense rows and nonoverlapping batch/head pitches required")
    strides = [t.stride() for t in (q, k, v, o, rel_h_w)]
    extent = _launch_envelope(batch, heads, length, dim, BLOCK_M, BLOCK_N, strides)
    for t in tensors:
        base, width = t.data_ptr(), t.element_size()
        if base < 0 or base % width or base + width * extent >= _UINT64_LIMIT:
            raise ValueError("formed byte addresses must be aligned and fit unsigned64")
    output_storage = o.untyped_storage()
    out_lo, out_hi = output_storage.data_ptr(), output_storage.data_ptr() + output_storage.nbytes()
    for t in (q, k, v, rel_h_w):
        storage = t.untyped_storage()
        lo, hi = storage.data_ptr(), storage.data_ptr() + storage.nbytes()
        if out_lo < hi and lo < out_hi:
            raise ValueError("caller output storage must be separate from all inputs")
    OUT_DTYPE = tl.float16 if q.dtype == torch.float16 else tl.bfloat16
    with torch.cuda.device(q.device):
        _fwd_kernel_aligned[(triton.cdiv(length, BLOCK_M), batch * heads, 1)](
            q, k, v, rel_h_w, float(sm_scale), o,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            rel_h_w.stride(0), rel_h_w.stride(1), rel_h_w.stride(2),
            batch, heads, length, 0, OUT_DTYPE=OUT_DTYPE,
            BIAS_LAST_SIZE=64, B0_NUMEL=128, BLOCK_DMODEL=dim,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=num_warps, num_stages=num_stages, enable_fp_fusion=True)

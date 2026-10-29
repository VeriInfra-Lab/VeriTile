"""Authored counterpart of the existing scalar-left FP32 score rewrite.

Derived from the unchanged attention_score_tile_checked.py and its independent
score-sum specification. Only the score/scale multiplication operands commute;
casts, branches, masks, loop, block pointers and stores are retained.
This is a new counterpart, not a recovered historical Python source.
"""
import triton
import triton.language as tl

@triton.jit
def _score_scalar_left_kernel(Q, K, M, sm_scale, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_oz, stride_oh, stride_on, Z, H, H_KV, N_CTX, ROUND_CTX, NKV_CTX, sliding_window_offset, sliding_window_size, SLIDING_WINDOW: tl.constexpr, COMPLEMENT_SLIDING_WINDOW: tl.constexpr, IS_EVEN_M: tl.constexpr, IS_EVEN_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    start_n = (tl.program_id(0)).to(tl.int64)
    off_hz = (tl.program_id(1)).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H//H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
    m_ptrs = M + off_hz * ROUND_CTX + tl.arange(0, BLOCK_M).to(tl.int64)
    o = tl.zeros([BLOCK_M], dtype=tl.float32)

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, NKV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, (start_n * BLOCK_N).to(tl.int32)),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )

    if IS_EVEN_N:
        k = tl.load(K_block_ptr)
    else:
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")


    lo = 0
    hi = tl.cast(N_CTX, tl.int64)
    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634   # 1/log(2)

    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        if IS_EVEN_M:
            q = tl.load(Q_block_ptr)
        else:
            q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

        m = tl.load(m_ptrs, mask=start_m + tl.arange(0, BLOCK_M) < N_CTX, other=0)

        # calc qk
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk_scale * qk

        if SLIDING_WINDOW:
            dist = (tl.arange(0, BLOCK_M)[:, None]).to(tl.int64) - (tl.arange(0, BLOCK_N)[None, :]).to(tl.int64) \
                 + start_m - start_n * BLOCK_N + tl.cast(sliding_window_offset, tl.int64)

            if COMPLEMENT_SLIDING_WINDOW:
                mask = (dist >= tl.cast(sliding_window_size, tl.int64))
            else:
                mask = (dist >= 0) & (dist < tl.cast(sliding_window_size, tl.int64))

        qk = qk - m[:, None]
        p = tl.math.exp2(qk) # (BLOCK_M, BLOCK_N)

        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)

        if not IS_EVEN_M:
            p = tl.where(
                ((tl.arange(0, BLOCK_M) + start_m) < N_CTX)[:, None],
                p, 0
            )

        o += tl.sum(p, axis=0)


        Q_block_ptr = tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))
        m_ptrs = m_ptrs + BLOCK_M

    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    o_range = tl.arange(0, BLOCK_N).to(tl.int64) + start_n * BLOCK_N # orange
    o_ptrs = Out + o_offset + o_range
    tl.store(o_ptrs, o.to(Out.type.element_ty), mask = o_range < NKV_CTX)

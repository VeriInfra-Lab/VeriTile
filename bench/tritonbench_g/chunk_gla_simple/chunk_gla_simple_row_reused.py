"""Derived from the repaired chunk_gla_simple JIT.

Source SHA256: ca368c049b1d8591ec4352873334fcdd79969150929b45c82fd8460249c5cd8e
Only row-index arange construction reuses the existing o_i vector; its
explicit INT64 widening and all floating operations remain.
"""
import triton
import triton.language as tl

@triton.jit
def chunk_simple_gla_fwd_kernel_o_row_reused(
    q,
    k,
    v,
    h,
    g,
    o,
    s_k_h,
    s_k_t,
    s_v_h,
    s_v_t,
    s_h_h,
    s_h_t,
    scale,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    # Kernel implementation
    # Widen before every address product; block-pointer offsets are only i32.
    i_v = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    s_k_h, s_k_t = tl.cast(s_k_h, tl.int64), tl.cast(s_k_t, tl.int64)
    s_v_h, s_v_t = tl.cast(s_v_h, tl.int64), tl.cast(s_v_t, tl.int64)
    s_h_h, s_h_t = tl.cast(s_h_h, tl.int64), tl.cast(s_h_t, tl.int64)

    o_i = tl.arange(0, BT)
    m_s = o_i[:, None] >= o_i[None, :]

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_s = tl.zeros([BT, BT], dtype=tl.float32)
    rows = i_t * BT + o_i.to(tl.int64)
    cols = i_v * BV + tl.arange(0, BV).to(tl.int64)
    # Keep range's signed induction bound valid across the uint32 constant interval.
    for i_k in range(tl.cast(tl.cdiv(K, BK), tl.int64)):
        keys = tl.cast(i_k, tl.int64) * BK + tl.arange(0, BK).to(tl.int64)
        p_q = q + i_bh * s_k_h + rows[:, None] * s_k_t + keys[None, :]
        p_k = k + i_bh * s_k_h + keys[:, None] + rows[None, :] * s_k_t
        p_h = h + i_bh * s_h_h + i_t * K * V + keys[:, None] * s_h_t + cols[None, :]
        b_q = tl.load(p_q, mask=(rows[:, None] < T) & (keys[None, :] < K), other=0)
        b_k = tl.load(p_k, mask=(keys[:, None] < K) & (rows[None, :] < T), other=0)
        b_h = tl.load(p_h, mask=(keys[:, None] < K) & (cols[None, :] < V), other=0)
        b_o += tl.dot(b_q, b_h, allow_tf32=False)
        b_s += tl.dot(b_q, b_k, allow_tf32=False)

    p_g = g + i_bh * T + rows
    b_g = tl.load(p_g, mask=rows < T, other=0)
    b_o = b_o * tl.exp(b_g)[:, None]
    b_s = b_s * tl.exp(b_g[:, None] - b_g[None, :])
    b_s = tl.where(m_s, b_s, 0)

    p_v = v + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    b_v = tl.load(p_v, mask=(rows[:, None] < T) & (cols[None, :] < V), other=0)
    b_o = (b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)) * scale
    p_o = o + i_bh * s_v_h + rows[:, None] * s_v_t + cols[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=(rows[:, None] < T) & (cols[None, :] < V))

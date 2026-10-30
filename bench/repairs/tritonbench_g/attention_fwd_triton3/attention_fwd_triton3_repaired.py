
import math
import torch
import triton
import triton.language as tl

_BLOCK_N = 64
_BLOCK_M = 64

@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,
                    K_block_ptr, V_block_ptr,
                    start_m, qk_scale, N_CTX,
                    sliding_window_offset, sliding_window_size,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, SLIDING_WINDOW: tl.constexpr,
                    IS_EVEN_M: tl.constexpr, IS_EVEN_N: tl.constexpr, COMPLEMENT_SLIDING_WINDOW: tl.constexpr
                ):
    start_m = start_m.to(tl.int64)
    # range of values handled by this stage
    if SLIDING_WINDOW and not COMPLEMENT_SLIDING_WINDOW:
        if COMPLEMENT_SLIDING_WINDOW:
            lo = 0
            hi = (((start_m + 1) * BLOCK_M + sliding_window_offset - sliding_window_size + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        else:
            lo = ((start_m * BLOCK_M + sliding_window_offset - sliding_window_size + 1) // BLOCK_N) * BLOCK_N
            hi = ((((start_m + 1) * BLOCK_M - 1) + sliding_window_offset + BLOCK_N) // BLOCK_N) * BLOCK_N
            if lo < 0:
                lo = tl.full((), 0, tl.int64)
            if hi > N_CTX:
                hi = N_CTX.to(tl.int64)

            lo = tl.multiple_of(lo, BLOCK_N)
            K_block_ptr = tl.advance(K_block_ptr, (0, lo.to(tl.int32)))
            V_block_ptr = tl.advance(V_block_ptr, (lo.to(tl.int32), 0))
    else:
        lo, hi = 0, N_CTX

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if IS_EVEN_N:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = qk * qk_scale

        if SLIDING_WINDOW:
            dist = tl.arange(0, BLOCK_M)[:, None] - tl.arange(0, BLOCK_N)[None, :] \
                   + start_m * BLOCK_M - start_n + sliding_window_offset

            if COMPLEMENT_SLIDING_WINDOW:
                mask = (dist >= sliding_window_size)
            else:
                mask = (dist >= 0) & (dist < sliding_window_size)

            qk = tl.where(mask, qk, float("-inf"))

        if not IS_EVEN_N:
            qk = tl.where(((tl.arange(0, BLOCK_N) + start_n) < N_CTX)[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)

        if not IS_EVEN_N:
            p = tl.where(((tl.arange(0, BLOCK_N) + start_n) < N_CTX)[None, :], p, 0)

        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        tmp = m_i - m_ij
        alpha_mask = (tmp != tmp)  # check nan
        alpha = tl.math.exp2(tmp)
        alpha = tl.where(alpha_mask, 1., alpha)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if IS_EVEN_N:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        acc += tl.dot(p.to(v.dtype), v)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    return acc, l_i, m_i


@triton.heuristics(
    {
        "IS_EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "IS_EVEN_N": lambda args: args["NKV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out, L,#
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, H_KV, #
              N_CTX,  #
              ROUND_CTX,
              NKV_CTX,
              sliding_window_offset,
              sliding_window_size,
              IS_EVEN_M: tl.constexpr,
              IS_EVEN_N: tl.constexpr,
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              END: tl.constexpr,
              INIT: tl.constexpr,
              SLIDING_WINDOW: tl.constexpr,
              COMPLEMENT_SLIDING_WINDOW: tl.constexpr
            ):

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    off_hkv = off_h // (H//H_KV)
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_hkv.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(NKV_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, NKV_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(ROUND_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # initialize pointer to m and l
    m_ptrs = M + off_hz * ROUND_CTX + offs_m
    l_ptrs = L + off_hz * ROUND_CTX + offs_m
    if INIT:
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    else:
        # don't have to check boundary for q len
        m_i = tl.load(m_ptrs).to(tl.float32)
        l_i = tl.load(l_ptrs).to(tl.float32)
        acc = tl.load(O_block_ptr).to(tl.float32)

    qk_scale = sm_scale
    qk_scale *= 1.4426950408889634   # 1/log(2)
    # load q: it will stay in SRAM throughout
    if IS_EVEN_M:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, #
                                    start_m, qk_scale, NKV_CTX, #
                                    sliding_window_offset, sliding_window_size,
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N,
                                    COMPLEMENT_SLIDING_WINDOW)
    # epilogue
    if (END):
        m_i += tl.math.log2(l_i)
        acc = acc / l_i[:, None]
    else:
        tl.store(l_ptrs, l_i)

    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

def _span(shape, strides):
    if not all(shape):
        return 0
    return 1 + sum((n - 1) * stride for n, stride in zip(shape, strides))


def _check_call(q, k, v, o, m, l, end, init, sliding_window,
                complement_sliding_window, bm, bn):
    tensors = (q, k, v, o, m, l)
    if any(type(t) is not torch.Tensor for t in tensors):
        raise TypeError("Q/K/V and supplied O/M/L must be ordinary tensors")
    if any(t.layout != torch.strided or not t.is_cuda or t.dtype != torch.float32
           or t.device != q.device or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("resolved strided CUDA FP32 tensors on one device required")
    if any(type(flag) is not bool for flag in (end, init, complement_sliding_window)):
        raise ValueError("END/INIT/complement flags must be bool")
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("rank-four Q and matching K/V required")
    z, h, nq, d = q.shape
    zk, hk, nk, dk = k.shape
    if z != zk or d != dk or d not in (16, 32, 64, 128) or h <= 0 or hk <= 0 or h % hk:
        raise ValueError("matching supported D and valid grouped-query heads required")
    if bm not in (16, 32, 64) or bn != bm:
        raise ValueError("persistent block sizes must be equal and in {16,32,64}")
    rounded = ((nq + 63) // 64) * 64
    if o.shape != (z, h, rounded, d) or m.shape != (z, h, rounded) or l.shape != m.shape:
        raise ValueError("O/M/L must expose the full 64-rounded workspace shapes")
    if not m.is_contiguous() or not l.is_contiguous():
        raise ValueError("M/L workspaces must be contiguous")
    if any(any(stride < 0 for stride in t.stride()) for t in (q, k, v, o)):
        raise ValueError("nonnegative source strides required")
    sz, sh, sr, sd = o.stride()
    if sd <= 0 or d * sd > sr or rounded * sr > sh or h * sh > sz:
        raise ValueError("output strides must separate every padded output element")
    if sliding_window is None:
        offset, size = 0, 0
    elif (isinstance(sliding_window, (tuple, list)) and len(sliding_window) == 2
          and all(type(x) is int for x in sliding_window)):
        offset, size = sliding_window
    else:
        raise ValueError("window must be None or a pair of signed32 integers")
    if not all(-(1 << 31) <= x < 1 << 31 for x in (offset, size)):
        raise ValueError("window metadata must fit signed32")
    if nq > (1 << 31) - 65 or nk > (1 << 31) - 65 or rounded >= 1 << 31 or z * h > 65536:
        raise ValueError("context and launch metadata exceed machine bounds")
    # This also bounds the initial K/V advance for empty windows. The source
    # keeps its original signed32 block-pointer offset ABI after the wide math.
    key_rows = max(nk + 64, max(0, rounded + offset - size + 1) + 64
                   if sliding_window is not None and not complement_sliding_window else nk + 64)
    if key_rows >= 1 << 31:
        raise ValueError("all initial and final K/V block offsets must fit signed32")
    formed = []
    for t, heads, rows in ((q, h, rounded + 64), (k, hk, key_rows),
                           (v, hk, key_rows), (o, h, rounded + 64)):
        s0, s1, s2, s3 = t.stride()
        formed.append(z*s0 + heads*s1 + rows*s2 + d*s3 + 1)
    formed.extend((z*h*rounded + rounded + 1,) * 2)
    intervals = []
    for t, extent in zip(tensors, formed):
        available = t.untyped_storage().nbytes() // 4 - t.storage_offset()
        if _span(t.shape, t.stride()) > available:
            raise ValueError("declared tensor view exceeds its live storage")
        base = t.data_ptr()
        if base < 0 or base % 4 or 4*extent >= 1 << 63 or base + 4*extent >= 1 << 64:
            raise ValueError("aligned formed addresses must fit the signed64/unsigned64 ABI")
        intervals.append((base // 4, base // 4 + available))
    for i, (lo, hi) in enumerate(intervals):
        for other_lo, other_hi in intervals[i+1:]:
            if not (hi <= other_lo or other_hi <= lo):
                raise ValueError("six live tensor intervals must be disjoint")


def _forward(
    q, k, v, sm_scale,
    o=None, m=None, l=None, end=False,
    sliding_window=None, init=False,
    complement_sliding_window=False
):
    global _BLOCK_N
    global _BLOCK_M
    _check_call(q, k, v, o, m, l, end, init, sliding_window,
                complement_sliding_window, _BLOCK_M, _BLOCK_N)
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]

    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}

    q_round_len = math.ceil(q.shape[2] / 64) * 64

    if sliding_window is not None:
        sliding_window_offset, sliding_window_size = sliding_window
    else:
        sliding_window_offset, sliding_window_size = None, None

    grid = lambda META: (
        triton.cdiv(q.shape[2], META["BLOCK_M"]),
        q.shape[0] * q.shape[1],
    )

    try:
        with torch.cuda.device(q.device):
            _attn_fwd[grid](
                q, k, v, sm_scale, m, o, l, #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1], k.shape[1], #
                q.shape[2],  #
                q_round_len,
                k.shape[2],
                sliding_window_offset,
                sliding_window_size,
                BLOCK_DMODEL=Lk,  #
                END=end,
                INIT=init,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                num_warps=4,
                num_stages=4
            )
    except triton.OutOfResources as E:
        _BLOCK_N = _BLOCK_N // 2
        _BLOCK_M = _BLOCK_M // 2
        _check_call(q, k, v, o, m, l, end, init, sliding_window,
                    complement_sliding_window, _BLOCK_M, _BLOCK_N)
        from warnings import warn
        warn(f"Triton Attention Output Resources. {E}\nUse smaller block size {_BLOCK_N}.")
        with torch.cuda.device(q.device):
            _attn_fwd[grid](
                q, k, v, sm_scale, m, o, l, #
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
                q.shape[0], q.shape[1], k.shape[1], #
                q.shape[2],  #
                q_round_len,
                k.shape[2],
                sliding_window_offset,
                sliding_window_size,
                BLOCK_DMODEL=Lk,  #
                END=end,
                INIT=init,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                num_warps=4,
                num_stages=4
            )

    if end:
        o = o[:, :, :q.shape[2], :].contiguous().to(q.dtype)

    return o, m, l




##################################################################################################################################################


import torch
import math

# Define the test function for _forward

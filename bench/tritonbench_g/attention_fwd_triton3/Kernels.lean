import VeriTile.Triton.DSL.GuardedJitFunctions
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 2000000

def innerSource : StatementJitFunction :=
  with_triton_params (acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, N_CTX, sliding_window_offset, sliding_window_size, BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N, COMPLEMENT_SLIDING_WINDOW) constexpr_args [BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N, COMPLEMENT_SLIDING_WINDOW] in triton {
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
                hi = tl.cast(N_CTX, tl.int64)

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
}

def signature : CallSignature where
  pointers := [("Q",.fp32),("K",.fp32),("V",.fp32),("M",.fp32),("Out",.fp32),("L",.fp32)]
  scalars := [("sm_scale",.fp32),("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("Z",.integer),("H",.integer),("H_KV",.integer),("N_CTX",.integer),("ROUND_CTX",.integer),("NKV_CTX",.integer),("sliding_window_offset",.integer),("sliding_window_size",.integer)]
  constexpr := ["IS_EVEN_M", "IS_EVEN_N", "BLOCK_M", "BLOCK_DMODEL", "BLOCK_N", "END", "INIT", "SLIDING_WINDOW", "COMPLEMENT_SLIDING_WINDOW"]
  readPorts := ["M","L","Out","Q","K","V"]
  writePorts := ["L","M","Out"]

def sourceKernel (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX : Nat)
    (sliding_window_offset sliding_window_size : Int)
    (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (IS_EVEN_M IS_EVEN_N END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Bool) : ComputeKernel :=
  with_triton_call signature call_args [sm_scale := sm_scale, stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, Z := Z, H := H, H_KV := H_KV, N_CTX := N_CTX, ROUND_CTX := ROUND_CTX, NKV_CTX := NKV_CTX, sliding_window_offset := sliding_window_offset, sliding_window_size := sliding_window_size]
    jit_functions [_attn_fwd_inner := innerSource] in triton {
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
}

def innerCSESource : StatementJitFunction :=
  with_triton_params (acc, l_i, m_i, q, K_block_ptr, V_block_ptr, start_m, qk_scale, N_CTX, sliding_window_offset, sliding_window_size, BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N, COMPLEMENT_SLIDING_WINDOW) constexpr_args [BLOCK_M, BLOCK_DMODEL, BLOCK_N, SLIDING_WINDOW, IS_EVEN_M, IS_EVEN_N, COMPLEMENT_SLIDING_WINDOW] in triton {
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
                hi = tl.cast(N_CTX, tl.int64)

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
            key_mask = (tl.arange(0, BLOCK_N) + start_n) < N_CTX
            qk = tl.where(key_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)

        if SLIDING_WINDOW:
            p = tl.where(mask, p, 0)

        if not IS_EVEN_N:
            p = tl.where(key_mask[None, :], p, 0)

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
  }

def kernelCSEKernel (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX : Nat)
    (sliding_window_offset sliding_window_size : Int)
    (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) (IS_EVEN_M IS_EVEN_N END INIT SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Bool) : ComputeKernel :=
  with_triton_call signature call_args [sm_scale := sm_scale, stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, Z := Z, H := H, H_KV := H_KV, N_CTX := N_CTX, ROUND_CTX := ROUND_CTX, NKV_CTX := NKV_CTX, sliding_window_offset := sliding_window_offset, sliding_window_size := sliding_window_size]
    jit_functions [_attn_fwd_inner_key_mask_cse := innerCSESource] in triton {
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

    acc, l_i, m_i = _attn_fwd_inner_key_mask_cse(acc, l_i, m_i, q, K_block_ptr, V_block_ptr, #
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
}

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast

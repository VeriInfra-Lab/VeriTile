import math
import struct

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L, M,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX, D0,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # initialize offsets for store
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    stride_qh_2d = stride_qh // stride_qm // stride_qk

    q_tile_ptr = tl.make_block_ptr(base=Q,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_qm, stride_qk),
                                   offsets=(
                                       off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    k_tile_ptr = tl.make_block_ptr(base=K,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_kn, stride_kk),
                                   offsets=(off_hz * stride_qh_2d, 0),
                                   block_shape=(BLOCK_N, BLOCK_DMODEL),
                                   order=(1, 0))
    v_tile_ptr = tl.make_block_ptr(base=V,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_vk, stride_vn),
                                   offsets=(off_hz * stride_qh_2d, 0),
                                   block_shape=(BLOCK_N, BLOCK_DMODEL),
                                   order=(1, 0))
    out_tile_ptr = tl.make_block_ptr(base=Out,
                                     shape=(D0, BLOCK_DMODEL),
                                     strides=(stride_om, stride_on),
                                     offsets=(off_hz * stride_qh_2d + start_m * BLOCK_M, 0),
                                     block_shape=(BLOCK_M, BLOCK_DMODEL),
                                     order=(1, 0))
    # load q: it will stay in SRAM throughout
    q = tl.load(q_tile_ptr)

    # loop over k, v and update accumulators
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_tile_ptr, boundary_check=(0, 1))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= (
            start_n + offs_n[None, :]), qk, float("-inf"))
        # compute new m
        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        # correct old l
        l_prev *= tl.exp(m_prev - m_curr)
        # attention weights
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        # rescale operands of matmuls
        l_rcp = 1. / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        # update acc
        p = p.to(tl.float16)
        v = tl.load(v_tile_ptr, boundary_check=(0, 1))
        acc += tl.dot(p.to(v.dtype), v)
        # update m_i and l_i
        l_prev = l_curr
        m_prev = m_curr
        # update pointers
        k_tile_ptr = tl.advance(k_tile_ptr, [BLOCK_N, 0])
        v_tile_ptr = tl.advance(v_tile_ptr, [BLOCK_N, 0])
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, l_prev)
    tl.store(m_ptrs, m_prev)

    acc = acc.to(tl.float16)
    tl.store(out_tile_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def _bwd_preprocess_shifted(
    Out, DO, L,
    NewDO, Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    tl.static_assert(BLOCK_M == 128)
    off_m = (tl.program_id(0) << 7) + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(NewDO + off_m[:, None] * D_HEAD + off_n[None, :], do)
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO,
    DQ, DK, DV,
    L, M,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H, N_CTX, D0,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # init tile_ptr
    stride_qz_2d = stride_qz // stride_qm // stride_qk
    stride_qh_2d = stride_qh // stride_qm // stride_qk

    q_tile_ptr = tl.make_block_ptr(base=Q,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_qm, stride_qk),
                                   offsets=(
                                       off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    k_tile_ptr = tl.make_block_ptr(base=K,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_kn, stride_kk),
                                   offsets=(
                                       off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    v_tile_ptr = tl.make_block_ptr(base=V,
                                   shape=(D0, BLOCK_DMODEL),
                                   strides=(stride_vk, stride_vn),
                                   offsets=(
                                       off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                   block_shape=(BLOCK_M, BLOCK_DMODEL),
                                   order=(1, 0))
    do_tile_ptr = tl.make_block_ptr(base=DO,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    dq_tile_ptr = tl.make_block_ptr(base=DQ,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    dk_tile_ptr = tl.make_block_ptr(base=DK,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    dv_tile_ptr = tl.make_block_ptr(base=DV,
                                    shape=(D0, BLOCK_DMODEL),
                                    strides=(stride_qm, stride_qk),
                                    offsets=(
                                        off_z * stride_qz_2d + off_h * stride_qh_2d, 0),
                                    block_shape=(BLOCK_M, BLOCK_DMODEL),
                                    order=(1, 0))
    # offset pointers for batch/head
    DQ += off_z * stride_qz + off_h * stride_qh
    for start_n in range(0, num_block):
        lo = start_n * BLOCK_M
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        dq_ptrs = DQ + (offs_qm[:, None] * stride_qm + offs_k[None, :] * stride_qk)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        m_ptrs = M + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_tile_ptr, boundary_check=(0, 1))
        v = tl.load(v_tile_ptr, boundary_check=(0, 1))
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q = tl.load(q_tile_ptr, boundary_check=(0, 1))
            # recompute p = softmax(qk, dim=-1).T
            # NOTE: `do` is pre-divided by `l`; no normalization here
            qk = tl.dot(q, tl.trans(k))
            qk = tl.where(offs_m_curr[:, None] >= (
                offs_n[None, :]), qk, float("-inf"))
            m = tl.load(m_ptrs + offs_m_curr)
            p = tl.exp(qk * sm_scale - m[:, None])
            # compute dv
            do = tl.load(do_tile_ptr, boundary_check=(0, 1))
            dv += tl.dot(tl.trans(p.to(tl.float16).to(do.dtype)), do)
            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(tl.float16).to(q.dtype)), q)
            # compute dq
            dq = tl.load(dq_tile_ptr)
            dq += tl.dot(ds.to(tl.float16).to(k.dtype), k)
            tl.store(dq_tile_ptr, dq)
            # increment pointers
            dq_ptrs += BLOCK_M * stride_qm
            q_tile_ptr = tl.advance(q_tile_ptr, [BLOCK_M, 0])
            do_tile_ptr = tl.advance(do_tile_ptr, [BLOCK_M, 0])
            dq_tile_ptr = tl.advance(dq_tile_ptr, [BLOCK_M, 0])
        q_tile_ptr = tl.advance(q_tile_ptr, [lo + (1 - num_block) * BLOCK_M, 0])
        do_tile_ptr = tl.advance(do_tile_ptr, [lo + (1 - num_block) * BLOCK_M, 0])
        dq_tile_ptr = tl.advance(dq_tile_ptr, [lo + (1 - num_block) * BLOCK_M, 0])
        # increment tile pointers
        k_tile_ptr = tl.advance(k_tile_ptr, [BLOCK_M, 0])
        v_tile_ptr = tl.advance(v_tile_ptr, [BLOCK_M, 0])
        # write-back
        tl.store(dv_tile_ptr, dv.to(tl.float16).to(DV.dtype.element_ty), boundary_check=(0, 1))
        tl.store(dk_tile_ptr, dk.to(tl.float16).to(DK.dtype.element_ty), boundary_check=(0, 1))
        dv_tile_ptr = tl.advance(dv_tile_ptr, [BLOCK_M, 0])
        dk_tile_ptr = tl.advance(dk_tile_ptr, [BLOCK_M, 0])


def _validate_inputs(q, k, v, sm_scale, *, allow_fp16=False):
    """Admit the whole-kernel CUDA domain before allocating or launching.

    The formal domain uses FP32; FP16 retains the original rounded arithmetic.
    The element-count bound protects all signed32 row/element expressions,
    including preprocessing and the backward negative block-pointer reset.
    Dereferenced byte displacements are below 4 * (2**31 - 1). Even the
    unused final dq_ptrs advance adds at most 128 * 128 * 4 bytes, hence
    all pointer displacements fit signed64.
    """
    tensors = (q, k, v)
    if not all(type(t) is torch.Tensor for t in tensors):
        raise TypeError("q, k and v must be tensors")
    if not all(t.layout == torch.strided and t.ndim == 4 for t in tensors):
        raise ValueError("q, k and v must be strided rank-4 tensors")
    if not (q.shape == k.shape == v.shape):
        raise ValueError("q, k and v must have the same [Z, H, N, D] shape")
    if not (q.device == k.device == v.device and q.device.type == "cuda"):
        raise ValueError("q, k and v must be on the same CUDA device")
    allowed_dtypes = (torch.float32, torch.float16) if allow_fp16 else (torch.float32,)
    if not (q.dtype == k.dtype == v.dtype and q.dtype in allowed_dtypes):
        raise ValueError("Q/K/V dtype is outside the selected checked or compatibility entry")
    if not all(t.is_contiguous() and not t.is_conj() and not t.is_neg() for t in tensors):
        raise ValueError("q, k and v must be contiguous with resolved conjugate/negative bits")
    z, h, n, d = q.shape
    if not (z > 0 and h > 0 and n > 0 and n % 128 == 0 and d in (16, 32, 64, 128)):
        raise ValueError("positive Z/H/N, N divisible by 128 and D in {16,32,64,128} required")
    # PyTorch contiguity ignores singleton strides; this kernel flattens heads
    # through stride_qh, so its address contract requires canonical strides.
    canonical_strides = (h * n * d, n * d, d, 1)
    if not all(t.stride() == canonical_strides for t in tensors):
        raise ValueError("q, k and v must have canonical contiguous strides")
    if q.numel() > 2**31 - 1 or z * h > 65535:
        raise ValueError("shape exceeds signed32 addressing or CUDA grid-y bounds")
    if type(sm_scale) not in (int, float):
        raise TypeError("sm_scale must be a real scalar")
    if type(sm_scale) is int and sm_scale >= 2**64:
        raise ValueError("integer sm_scale must fit the Triton scalar ABI")
    try:
        scale = float(sm_scale)
    except OverflowError as error:
        raise ValueError("sm_scale must be representable as a finite real scalar") from error
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("sm_scale must be finite and positive")
    try:
        launch_scale = struct.unpack("f", struct.pack("f", scale))[0]
    except OverflowError as error:
        raise ValueError("sm_scale must remain finite and positive in FP32") from error
    if not (math.isfinite(launch_scale) and launch_scale > 0):
        raise ValueError("sm_scale must remain finite and positive in FP32")
    return sm_scale


def _pointer_contract(tensors, elements):
    # Source final advancement plus the pinned profiles' masked prefetch reserve.
    envelope = elements + 65536
    for tensor in tensors:
        pointer, width = tensor.data_ptr(), tensor.element_size()
        if tensor.storage_offset() < 0 or width*(tensor.storage_offset()+tensor.numel()) > tensor.untyped_storage().nbytes():
            raise ValueError("source tensor extent must be backed by its actual storage")
        if pointer < 0 or pointer % width or pointer + width*(envelope + 1) >= 2**64:
            raise ValueError("naturally aligned formed source byte addresses must fit unsigned64")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        return _attention._forward(ctx, q, k, v, sm_scale, allow_fp16=False)

    @staticmethod
    def _forward(ctx, q, k, v, sm_scale, *, allow_fp16):
        sm_scale = _validate_inputs(q, k, v, sm_scale, allow_fp16=allow_fp16)
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.float32)
        m = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        D0 = q.shape[0] * q.shape[1] * q.shape[2]
        _pointer_contract((q, k, v, o, L, m), q.numel())
        with torch.cuda.device(q.device):
            _fwd_kernel[grid](
                q, k, v, sm_scale,
                L, m,
                o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                q.shape[0], q.shape[1], q.shape[2], D0,
                BLOCK_M=BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=Lk, num_warps=num_warps,
                num_stages=2,
            )

        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        q, k, v, o, l, m = ctx.saved_tensors
        if type(do) is not torch.Tensor:
            raise TypeError("incoming gradient must be a tensor")
        if not (do.layout == torch.strided and do.shape == o.shape and
                do.device == o.device and do.dtype == o.dtype and
                not do.is_conj() and not do.is_neg()):
            raise ValueError("incoming gradient must match output shape/device/dtype with resolved bits")
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        D0 = q.shape[0] * q.shape[1] * q.shape[2]
        _pointer_contract((q, k, v, o, l, m, do, dq, dk, dv, do_scaled, delta), q.numel())
        with torch.cuda.device(q.device):
            _bwd_preprocess_shifted[(ctx.grid[0] * ctx.grid[1], )](
                o, do, l,
                do_scaled, delta,
                BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
            )
            _bwd_kernel[(ctx.grid[1],)](
                q, k, v, ctx.sm_scale,
                o, do_scaled,
                dq, dk, dv,
                l, m,
                delta,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                q.shape[0], q.shape[1], q.shape[2], D0,
                ctx.grid[0],
                BLOCK_M=BLOCK, BLOCK_N=BLOCK,
                BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
                num_stages=1,
            )
        return dq, dk, dv, None


attention = _attention.apply


class _attention_compatibility(torch.autograd.Function):
    """Retained FP16 compatibility path; the checked native theorem selects FP32."""
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
        return _attention._forward(ctx, q, k, v, sm_scale, allow_fp16=True)

    backward = staticmethod(_attention.backward)


attention_compatibility = _attention_compatibility.apply

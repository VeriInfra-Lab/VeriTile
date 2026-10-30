
import torch
import triton
import triton.language as tl

@triton.jit
def parallel_rebased_fwd_kernel(
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    o,  # output [B, H, L, D_head_V]
    z,  # normalizer [B, H, L]
    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1
    scale,  # D_head_K ** -0.5
    B,  # batch size
    H,  # H
    T,  # T
    K: tl.constexpr,  # D_head_K
    V: tl.constexpr,  # D_head_V
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K/V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
):
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_z, b_z.to(p_z.dtype.element_ty),
                 mask=((i_c * BTL + tl.arange(0, BTL)) < T))

@triton.jit
def _parallel_rebased_bwd_dq(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dq,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                             (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c * BTL + tl.arange(0, BTL)) < T, other=0)

    for _ in range(0, i_c * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        # [BQ, BD]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        # [BTL, BK]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_k_h, (T, K),
                             (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return

@triton.jit
def _parallel_rebased_bwd_dkv(
    i_bh, i_c, i_k, i_v, i_h,
    q, k, v, do, dz, dk, dv, s_k_h, s_k_t, s_k_d, s_v_h,
    s_v_t, s_v_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # compute dk dv
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K), (s_k_t, s_k_d),
                            (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                            (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    b_k, b_v = tl.load(p_k, boundary_check=(0, 1), padding_option="zero"), tl.load(
        p_v, boundary_check=(0, 1), padding_option="zero")
    b_dk, b_dv = tl.zeros([BTL, BK], dtype=tl.float32), tl.zeros(
        [BTL, BV], dtype=tl.float32)

    for i in range((tl.cdiv(T, BTS) * BTS)-BTS, (i_c + 1) * BTL - BTS, -BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")  # [BK, BTS]
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)  # [BV, BTS]
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T, other=0)
        b_s = tl.dot(b_k.to(b_q.dtype), b_q, allow_tf32=False) * \
            scale  # [BTL, BTS]
        # Padded future query lanes must not contribute even for non-finite values.
        valid_query = (i + tl.arange(0, BTS)) < T
        b_s = tl.where(valid_query[None, :], b_s, 0)
        b_s2 = b_s * b_s
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_ds = tl.dot(b_v, b_do, allow_tf32=False) * scale
        if i_v == 0:
            b_ds += b_dz[None, :] * scale
        else:
            b_ds = b_ds
        b_ds = tl.where(valid_query[None, :], b_ds, 0)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype),
                       tl.trans(b_q), allow_tf32=False)

    tl.debug_barrier()
    o_q, o_k = tl.arange(0, BTS), tl.arange(0, BTL)
    for i in range(i_c*BTL, (i_c+1)*BTL, BTS):
        p_q = tl.make_block_ptr(
            q + i_bh * s_k_h, (K, T), (s_k_d, s_k_t), (i_k * BK, i), (BK, BTS), (0, 1))
        p_do = tl.make_block_ptr(
            do + i_bh * s_v_h, (V, T), (s_v_d, s_v_t), (i_v * BV, i), (BV, BTS), (0, 1))
        p_dz = dz + i_bh * T + i + tl.arange(0, BTS)
        b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")  # [BD, BQ]
        b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)
        b_dz = tl.load(p_dz, mask=(i + tl.arange(0, BTS)) < T, other=0)
        # [BK, BQ]
        m_s = (o_k[:, None] <= o_q[None, :]) & ((i + tl.arange(0, BTS))[None, :] < T)
        b_s = tl.dot(b_k, b_q, allow_tf32=False) * scale
        b_s2 = b_s * b_s
        b_s = tl.where(m_s, b_s, 0)
        b_s2 = tl.where(m_s, b_s2, 0)

        b_ds = tl.dot(b_v, b_do, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[None, :]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        # [BK, BD]
        b_dv += tl.dot(b_s2.to(b_q.dtype), tl.trans(b_do), allow_tf32=False)
        b_dk += tl.dot((2 * b_ds * b_s).to(b_q.dtype),
                       tl.trans(b_q), allow_tf32=False)
        o_q += BTS

    p_dk = tl.make_block_ptr(dk + (i_bh + B * H * i_v) * s_k_h,
                             (T, K), (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (i_bh + B * H * i_k) * s_v_h,
                             (T, V), (s_v_t, s_v_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))
    return

@triton.jit
def parallel_rebased_bwd_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_rebased_bwd_dq(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d,
        scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )

def _check_dense_fp32_inputs(q, k, v):
    """Checked dense, resolved FP32 CUDA tensors; no Q/K/V materialization."""
    if any(type(x) is not torch.Tensor for x in (q, k, v)):
        raise TypeError("expected ordinary Torch tensors")
    if any(x.layout != torch.strided or not x.is_cuda for x in (q, k, v)):
        raise ValueError("expected strided CUDA tensors")
    if any(x.is_neg() or x.is_conj() for x in (q, k, v)):
        raise ValueError("expected resolved tensor views")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected rank-four Q, K and V")
    if q.shape != k.shape or v.shape[:3] != k.shape[:3]:
        raise ValueError("expected Q/K=[B,H,T,K] and V=[B,H,T,V]")
    if not (1 <= k.shape[-1] <= 128) or v.shape[-1] < 1:
        raise ValueError("expected 1 <= K <= 128 and V >= 1")
    if not (q.dtype == k.dtype == v.dtype == torch.float32):
        raise ValueError("this revised attention contract requires FP32 inputs")
    if q.device != k.device or q.device != v.device:
        raise ValueError("attention inputs must share one device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("this revised attention contract requires contiguous inputs")

    for x in (q, k, v):
        if x.numel():
            expected_stride = 1
            for size, stride in zip(reversed(x.shape), reversed(x.stride())):
                if size > 1 and stride != expected_stride:
                    raise ValueError("active dimensions require canonical dense strides")
                expected_stride *= size
            storage = x.untyped_storage()
            base, capacity, pointer = storage.data_ptr(), storage.nbytes(), x.data_ptr()
            if (pointer % 4 or pointer < base or
                    pointer + 4 * x.numel() > base + capacity or
                    base + capacity > 2**63):
                raise ValueError("tensor elements exceed the FP32 storage span")

    B, H, T, K = k.shape
    V = v.shape[-1]
    # Empty tensors take the explicit no-launch path below.
    if B == 0 or H == 0 or T == 0:
        return
    BV = max(16, min(128, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    if B * H > 65535 or triton.cdiv(T, 64) > 65535:
        raise ValueError("attention launch y/z axes exceed 65535")
    # Includes scratch sharding, inactive lanes and final pointer advances.
    formed_limit = (NV * B * H * T + 256) * max(K, V) + 128
    if formed_limit >= 2**31:
        raise ValueError("attention formed indices exceed signed 32-bit range")
    if any(x.data_ptr() + 4 * formed_limit >= 2**63 for x in (q, k, v)):
        raise ValueError("attention formed pointers exceed signed 64-bit range")


def _check_cotangent_storage(x, formed_limit=None):
    if any(stride < 0 for stride in x.stride()):
        raise ValueError("cotangent strides must be nonnegative")
    if x.numel():
        top = sum((size - 1) * stride for size, stride in zip(x.shape, x.stride()))
        storage = x.untyped_storage()
        base, capacity, pointer = storage.data_ptr(), storage.nbytes(), x.data_ptr()
        if (pointer % 4 or pointer < base or pointer + 4 * (top + 1) > base + capacity or
                base + capacity > 2**63):
            raise ValueError("cotangent elements exceed their FP32 storage span")
        if formed_limit is not None and pointer + 4 * formed_limit >= 2**63:
            raise ValueError("cotangent formed pointers exceed signed 64-bit range")


def _check_backward_inputs(q, k, v, do, dz):
    _check_dense_fp32_inputs(q, k, v)
    if any(type(x) is not torch.Tensor for x in (do, dz)):
        raise TypeError("expected ordinary Torch cotangent tensors")
    if any(x.layout != torch.strided or not x.is_cuda or x.dtype != torch.float32
           for x in (do, dz)):
        raise ValueError("expected strided FP32 CUDA cotangents")
    if any(x.is_neg() or x.is_conj() for x in (do, dz)):
        raise ValueError("expected resolved cotangent views")
    if do.device != q.device or dz.device != q.device:
        raise ValueError("cotangents must share the saved inputs' device")
    if do.shape != v.shape or dz.shape != q.shape[:3]:
        raise ValueError("expected dO=[B,H,T,V] and dZ=[B,H,T]")
    for x in (do, dz):
        _check_cotangent_storage(x)


def _check_prepared_cotangents(q, k, v, do, dz):
    _check_backward_inputs(q, k, v, do, dz)
    B, H, T, K = k.shape
    V = v.shape[-1]
    BV = max(16, min(128, triton.next_power_of_2(V)))
    NV = triton.cdiv(V, BV)
    formed_limit = (NV * B * H * T + 256) * max(K, V) + 128
    for x in (do, dz):
        if not x.is_contiguous():
            raise ValueError("prepared cotangents must be contiguous")
        if x.numel():
            expected_stride = 1
            for size, stride in zip(reversed(x.shape), reversed(x.stride())):
                if size > 1 and stride != expected_stride:
                    raise ValueError("prepared cotangents require canonical active strides")
                expected_stride *= size
        _check_cotangent_storage(x, formed_limit)


class ParallelBasedFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, scale):
        _check_dense_fp32_inputs(q, k, v)
        with torch.cuda.device(q.device):
            B, H, T, K, V = *k.shape, v.shape[-1]
            if B == 0 or H == 0 or T == 0:
                ctx.save_for_backward(q, k, v)
                ctx.scale = scale
                return (torch.empty((B, H, T, V), dtype=q.dtype, device=q.device),
                        torch.empty((B, H, T), dtype=q.dtype, device=q.device))
            BTL, BTS = 128, 32
            assert BTL % BTS == 0
            BK = min(128, triton.next_power_of_2(k.shape[-1]))
            BV = min(128, triton.next_power_of_2(v.shape[-1]))
            BK, BV = max(BK, 16), max(BV, 16)
            B, H, T, K, V = *k.shape, v.shape[-1]
            num_stages = 2
            num_warps = 4
            NK = triton.cdiv(K, BK)
            NV = triton.cdiv(V, BV)
            grid = (NK * NV, triton.cdiv(T, BTL), B * H)

            assert NK == 1, "will encounter some synchronization issue if not."

            o = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)
            z = torch.empty(NK, B, H, T, dtype=q.dtype, device=q.device)
            parallel_rebased_fwd_kernel[grid](
                q, k, v, o, z,
                T * K, K, 1,
                T * V, V, 1,
                scale,
                B=B, H=H, T=T, K=K, V=V,
                BTL=BTL, BTS=BTS, BK=BK, BV=BV,
                num_warps=num_warps,
                num_stages=num_stages
            )
            ctx.save_for_backward(q, k, v)
            ctx.scale = scale
            return o.sum(0).to(q.dtype), z.sum(0).to(q.dtype)

    @staticmethod
    def backward(ctx, do, dz):
        q, k, v = ctx.saved_tensors
        scale = ctx.scale
        _check_backward_inputs(q, k, v, do, dz)
        with torch.cuda.device(q.device):
            B, H, T, K, V = *k.shape, v.shape[-1]
            if B == 0 or H == 0 or T == 0:
                return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v), None
            # Ordinary autograd cotangents may be expanded views (e.g. sum loss).
            do, dz = do.contiguous(), dz.contiguous()
            _check_prepared_cotangents(q, k, v, do, dz)
            BTL, BTS = 64, 32
            assert BTL % BTS == 0
            BK = min(128, triton.next_power_of_2(k.shape[-1]))
            BV = min(128, triton.next_power_of_2(v.shape[-1]))
            BK, BV = max(BK, 16), max(BV, 16)
            B, H, T, K, V = *k.shape, v.shape[-1]
            num_stages = 2
            num_warps = 4
            NK = triton.cdiv(K, BK)
            NV = triton.cdiv(V, BV)
            grid = (NK * NV, triton.cdiv(T, BTL), B * H)

            assert NK == 1, "will encounter some synchronization issue if not"

            dq = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
            dk = torch.empty(NV, B, H, T, K, dtype=q.dtype, device=q.device)
            dv = torch.empty(NK, B, H, T, V, dtype=q.dtype, device=q.device)

            parallel_rebased_bwd_kernel[grid](
                q, k, v, do, dz, dq, dk, dv,
                T * K, K, 1,
                T * V, V, 1,
                scale,
                B=B, H=H, T=T, K=K, V=V,
                BTL=BTL, BTS=BTS, BK=BK, BV=BV,
                num_warps=num_warps,
                num_stages=num_stages
            )

            return dq.sum(0).to(q.dtype), dk.sum(0).to(k.dtype), dv.sum(0).to(v.dtype), None

triton_parallel_based = ParallelBasedFunction.apply

def parallel_rebased(q, k, v, eps=1e-5, use_scale=True, use_normalize=True, return_both=False):
    """Dense FP32 causal quadratic attention; return_both overrides normalization."""
    _check_dense_fp32_inputs(q, k, v)
    if any(type(flag) is not bool for flag in (use_scale, use_normalize, return_both)):
        raise TypeError("attention flags must be Python bools")
    if type(eps) not in (int, float):
        raise TypeError("eps must be a Python int or float")
    try:
        eps = float(eps)
    except OverflowError as error:
        raise ValueError("eps must convert to a Python float") from error
    with torch.cuda.device(q.device):
        assert q.shape[-1] <= 128, "only support feature dim up to 128"
        if use_scale:
            scale = q.shape[-1] ** -0.5
        else:
            scale = 1
        o, z = triton_parallel_based(q, k, v, scale)
        if return_both:
            return o, z
        if use_normalize:
            o = o / (z[..., None] + eps)
        else:
            o = o
        return o.to(q.dtype)


# Strict fusion candidates: remove one intermediate score assignment.
@triton.jit
def parallel_rebased_fwd_fused_kernel(
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_V]
    v,  # value [B, H, L, D_head_V]
    o,  # output [B, H, L, D_head_V]
    z,  # normalizer [B, H, L]
    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1
    scale,  # D_head_K ** -0.5
    B,  # batch size
    H,  # H
    T,  # T
    K: tl.constexpr,  # D_head_K
    V: tl.constexpr,  # D_head_V
    BTL: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BTS: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K/V
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
):
    # i_c: chunk index. used for sequence parallelism
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (T, K), (s_qk_t, s_qk_d), (i_c * BTL, i_k * BK), (BTL, BK), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, 0), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (0, i_v * BV), (BTS, BV), (1, 0))

    # [BQ, BD] block Q, in the shared memory throughout the whole kernel
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_q = (b_q * scale).to(b_q.dtype)
    b_o = tl.zeros([BTL, BV], dtype=tl.float32)
    b_z = tl.zeros([BTL], dtype=tl.float32)

    # Q block and K block have no overlap
    # no need for mask, thereby saving flops
    for _ in range(0, i_c * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")

        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)
        b_s = b_s * b_s
        b_z += tl.sum(b_s, axis=1)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))

    # # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BTL)
    # # sync threads, easy for compiler to optimize
    # tl.debug_barrier()

    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (K, T), (s_qk_d, s_qk_t), (i_k * BK, i_c * BTL), (BK, BTS), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c * BTL, i_v * BV), (BTS, BV), (1, 0))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BK, BTS]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BTS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s * b_s, 0)
        b_z += tl.sum(b_s, axis=1)
        # [BTL, BV]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)
        p_k = tl.advance(p_k, (0, BTS))
        p_v = tl.advance(p_v, (BTS, 0))
        o_k += BTS

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (T, V), (s_vo_t, s_vo_d), (i_c*BTL, i_v*BV), (BTL, BV), (1, 0))
    p_z = z + (i_bh + B * H * i_k) * T + i_c * BTL + tl.arange(0, BTL)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    if i_v == 0:
        tl.store(p_z, b_z.to(p_z.dtype.element_ty),
                 mask=((i_c * BTL + tl.arange(0, BTL)) < T))

@triton.jit
def _parallel_rebased_bwd_dq_fused(
    i_bh,
    i_c,
    i_k,
    i_v,
    i_h,
    q,
    k,
    v,
    do,
    dz,
    dq,
    s_k_h,
    s_k_t,
    s_k_d,
    s_v_h,
    s_v_t,
    s_v_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    p_do = tl.make_block_ptr(do + i_bh * s_v_h, (T, V), (s_v_t, s_v_d),
                             (i_c * BTL, i_v * BV), (BTL, BV), (1, 0))
    p_q = tl.make_block_ptr(q + (i_bh) * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1), padding_option="zero")
    b_do = tl.load(p_do, boundary_check=(0, 1), padding_option="zero").to(b_q.dtype)
    b_q = (b_q * scale).to(b_q.dtype)
    b_dq = tl.zeros([BTL, BK], dtype=tl.float32)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (0, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, 0), (BV, BTS), (0, 1))
    p_dz = dz + i_bh * T + i_c * BTL + tl.arange(0, BTL)
    b_dz = tl.load(p_dz, mask=(i_c * BTL + tl.arange(0, BTL)) < T, other=0)

    for _ in range(0, i_c * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_s = tl.dot(b_q, tl.trans(b_k), allow_tf32=False)
        # [BQ, BD]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_v.dtype), b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))

    b_dq *= scale
    o_q = tl.arange(0, BTL)
    o_k = tl.arange(0, BTS)
    p_k = tl.make_block_ptr(k + i_bh * s_k_h, (T, K),
                            (s_k_t, s_k_d), (i_c * BTL, i_k * BK), (BTS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + i_bh * s_v_h, (V, T),
                            (s_v_d, s_v_t), (i_v * BV, i_c * BTL), (BV, BTS), (0, 1))
    # Q block and K block have overlap. masks required
    for _ in range(i_c * BTL, (i_c + 1) * BTL, BTS):
        # [BTS, BK]
        b_k = tl.load(p_k, boundary_check=(0, 1), padding_option="zero")
        # [BV, BTS]
        b_v = tl.load(p_v, boundary_check=(0, 1), padding_option="zero")
        # [BTL, BTS]
        m_s = o_q[:, None] >= o_k[None, :]
        b_ds = tl.dot(b_do, b_v, allow_tf32=False)
        if i_v == 0:
            b_ds += b_dz[:, None]
        else:
            b_ds = b_ds
        b_ds = tl.where(m_s, b_ds, 0) * scale
        b_s = tl.where(m_s, tl.dot(b_q, tl.trans(b_k), allow_tf32=False), 0)
        # [BTL, BK]
        b_dq += tl.dot((2 * b_ds * b_s).to(b_k.dtype),
                       b_k, allow_tf32=False)
        p_k = tl.advance(p_k, (BTS, 0))
        p_v = tl.advance(p_v, (0, BTS))
        o_k += BTS
    p_dq = tl.make_block_ptr(dq + (i_bh + B * H * i_v) * s_k_h, (T, K),
                             (s_k_t, s_k_d), (i_c*BTL, i_k*BK), (BTL, BK), (1, 0))
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))
    return

@triton.jit
def parallel_rebased_bwd_fused_kernel(
    q,
    k,
    v,
    do,
    dz,
    dq,
    dk,
    dv,
    s_qk_h,
    s_qk_t,
    s_qk_d,
    s_vo_h,
    s_vo_t,
    s_vo_d,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BTL: tl.constexpr,
    BTS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr
):
    i_kv, i_c, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(V, BV)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)
    i_h = i_bh % H
    _parallel_rebased_bwd_dq_fused(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dq, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d, scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )
    tl.debug_barrier()
    _parallel_rebased_bwd_dkv(
        i_bh, i_c, i_k, i_v, i_h,
        q, k, v, do, dz, dk, dv, s_qk_h, s_qk_t, s_qk_d, s_vo_h,
        s_vo_t, s_vo_d,
        scale,
        B=B, H=H, T=T, K=K, V=V, BTL=BTL, BTS=BTS, BK=BK, BV=BV
    )

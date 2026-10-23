
import math
import torch
import triton
import triton.language as tl

_BLOCK_N = 64
_BLOCK_M = 64

@triton.heuristics(
    {
        "IS_EVEN_M": lambda args: args["N_CTX"] % args["BLOCK_M"] == 0,
        "IS_EVEN_N": lambda args: args["NKV_CTX"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _score_kernel(
    Q, K, M, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,  #
    stride_kz, stride_kh, stride_kn, stride_kk,  #
    stride_oz, stride_oh, stride_on,
    Z, H, H_KV, #
    N_CTX,  #
    ROUND_CTX,
    NKV_CTX,
    sliding_window_offset,
    sliding_window_size,
    SLIDING_WINDOW: tl.constexpr,
    COMPLEMENT_SLIDING_WINDOW: tl.constexpr,
    IS_EVEN_M: tl.constexpr,
    IS_EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,  #
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
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
        qk = qk * qk_scale

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


def _check_launch_domain(z, heads, kv_heads, n, r, nk, d, block):
    """Representation bounds for a nonempty launch, including masked pointers.

    Triton block-pointer offsets/advances remain signed32. Other source index
    expressions are widened before arithmetic. Displacements are FP32 bytes;
    this guard is independent of the allocator's ordinary valid-base contract.
    """
    i32_max, i64_max = (1 << 31) - 1, (1 << 63) - 1
    metadata = (z, heads, kv_heads, n, r, nk, d,
                heads * n * d, n * d, kv_heads * nk * d, nk * d,
                heads * nk)
    if not all(0 <= value <= i32_max for value in metadata):
        raise ValueError("launch shapes and dense strides must fit signed int32")
    # The wrapper checks each attempted block, including the selected retry.
    q_pad = triton.cdiv(n, block) * block
    k_pad = triton.cdiv(nk, block) * block
    if q_pad + block > i32_max or k_pad > i32_max:
        raise ValueError("padded block coordinates and final advance must fit signed int32")
    if triton.cdiv(nk, block) > i32_max or z * heads > 65535:
        raise ValueError("launch grid exceeds the CUDA grid representation")
    # End-exclusive formation bounds include the final Q/M advance and its
    # masked lanes. K/O include every masked key-tail lane.
    extents = ((z * heads - 1) * n * d + (q_pad + block) * d,
               (z * kv_heads - 1) * nk * d + k_pad * d,
               (z * heads - 1) * r + q_pad + block,
               (z * heads - 1) * nk + k_pad,
               z * heads * r)
    if not all(0 <= value and 4 * value <= i64_max for value in extents):
        raise ValueError("padded FP32 byte displacements must fit signed int64")
    return extents[:2] + (max(extents[2], extents[4]), extents[3])


def _check_pointer_domain(tensors, extents):
    for tensor, extent in zip(tensors, extents):
        base = tensor.data_ptr()
        if base % 4 or base < 0 or base + 4 * extent >= (1 << 64):
            raise ValueError("formed FP32 addresses must fit aligned unsigned int64")

def get_score(q, k, m, sliding_window, complement_sliding_window):
    global _BLOCK_N
    global _BLOCK_M

    if any(type(t) is not torch.Tensor for t in (q, k, m)):
        raise TypeError("ordinary tensors required")
    if q.ndim != 4 or k.ndim != 4 or m.ndim != 3:
        raise ValueError("get_score requires Q/K rank 4 and M rank 3")
    for tensor in (q, k, m):
        if tensor.dtype != torch.float32 or not tensor.is_cuda:
            raise ValueError("get_score requires CUDA FP32 tensors")
        if tensor.device != q.device:
            raise ValueError("get_score requires tensors on one CUDA device")
    z, heads, n, d = q.shape
    if k.shape[0] != z or k.shape[3] != d:
        raise ValueError("Q and K must have matching batch and feature dimensions")
    kv_heads = k.shape[1]
    if kv_heads == 0 or heads % kv_heads != 0:
        raise ValueError("KV heads must be positive and divide query heads")
    if m.shape[:2] != (z, heads) or m.shape[2] < n:
        raise ValueError("M must provide one normalizer for every query row")
    if d < 16 or d & (d - 1):
        raise ValueError("feature dimension must be a power of two at least 16")
    if _BLOCK_M != _BLOCK_N or _BLOCK_M not in (16, 32, 64):
        raise ValueError("query/key blocks must be equal and one of 16, 32, 64")
    if not isinstance(complement_sliding_window, bool):
        raise ValueError("complement_sliding_window must be a bool")
    if sliding_window is not None:
        if not isinstance(sliding_window, (tuple, list)) or len(sliding_window) != 2:
            raise ValueError("sliding_window must be an offset/size pair")
        if not all(isinstance(value, int) for value in sliding_window):
            raise ValueError("sliding_window offset and size must be integers")
        if not all(-(1 << 31) <= value < (1 << 31) for value in sliding_window):
            raise ValueError("sliding_window offset and size must fit signed int32")

    if any(t.layout != torch.strided or not t.is_contiguous() for t in (q, k, m)):
        raise ValueError("get_score requires contiguous strided tensors")
    if any(t.is_neg() or t.is_conj() for t in (q, k, m)):
        raise ValueError("get_score requires resolved logical tensor values")
    N_CTX = q.size(-2)
    NKV_CTX = k.size(-2)
    ROUND_CTX = m.size(-1)
    ret = torch.zeros(
        (q.size(0), q.size(1), k.size(2)),
        dtype=k.dtype, device=k.device
    )
    if N_CTX == 0 or ret.numel() == 0:
        return ret
    if sliding_window is not None:
        sliding_window_offset, sliding_window_size = sliding_window
    else:
        sliding_window_offset, sliding_window_size = None, None

    grid = lambda META: (
        triton.cdiv(k.shape[2], META["BLOCK_N"]),
        q.shape[0] * q.shape[1]
    )
    sm_scale = 1 / math.sqrt(q.size(-1))

    with torch.cuda.device(q.device):
        try:
            extents = _check_launch_domain(z, heads, kv_heads, n, ROUND_CTX, NKV_CTX, d, _BLOCK_M)
            _check_pointer_domain((q, k, m, ret), extents)
            _score_kernel[grid](
                q, k, m, sm_scale, ret,
                heads * n * d, n * d, d, 1,
                kv_heads * NKV_CTX * d, NKV_CTX * d, d, 1,
                heads * NKV_CTX, NKV_CTX, 1,
                q.size(0), q.size(1), k.size(1),
                N_CTX, ROUND_CTX, NKV_CTX,
                sliding_window_offset,
                sliding_window_size,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_DMODEL=q.size(-1),
                enable_fp_fusion=True
            )
        except triton.OutOfResources as E:
            from warnings import warn
            _BLOCK_N = max(16, _BLOCK_N // 2)
            _BLOCK_M = max(16, _BLOCK_M // 2)
            warn(f"Triton Attention Output Resources. {E}\nUse smaller block size {_BLOCK_N}.")
            extents = _check_launch_domain(z, heads, kv_heads, n, ROUND_CTX, NKV_CTX, d, _BLOCK_M)
            _check_pointer_domain((q, k, m, ret), extents)
            _score_kernel[grid](
                q, k, m, sm_scale, ret,
                heads * n * d, n * d, d, 1,
                kv_heads * NKV_CTX * d, NKV_CTX * d, d, 1,
                heads * NKV_CTX, NKV_CTX, 1,
                q.size(0), q.size(1), k.size(1),
                N_CTX, ROUND_CTX, NKV_CTX,
                sliding_window_offset,
                sliding_window_size,
                SLIDING_WINDOW=(sliding_window is not None),
                COMPLEMENT_SLIDING_WINDOW=complement_sliding_window,
                BLOCK_M=_BLOCK_M,
                BLOCK_N=_BLOCK_N,
                BLOCK_DMODEL=q.size(-1),
                enable_fp_fusion=True
            )

    return ret

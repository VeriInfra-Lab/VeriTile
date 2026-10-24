
import math
import torch
import triton
import triton.language as tl

def _storage(t, envelope, name):
    if t.storage_offset() < 0 or 4 * (t.storage_offset() + t.numel()) > t.untyped_storage().nbytes():
        raise ValueError(f"{name}: complete storage extent")
    if t.data_ptr() % 4:
        raise ValueError(f"{name}: four-byte alignment")
    if t.untyped_storage().data_ptr() + t.untyped_storage().nbytes() > 2**64:
        raise ValueError(f"{name}: pointer backing")
    if t.data_ptr() + 4 * envelope >= 2**64:
        raise ValueError(f"{name}: formed pointer envelope")


def _validate(q, k, v, seqlens, block_count, block_offset, column_count,
              column_index, scale, bm, bn):
    if type(bm) is not int or type(bn) is not int or bm != 64 or bn != 64:
        raise ValueError("selected 64-by-64 tile specialization required")
    if type(scale) not in (int, float) or not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38:
        raise ValueError("finite FP32-range scale required")
    for t in (q, k, v):
        if not isinstance(t, torch.Tensor) or t.ndim != 4 or t.dtype != torch.float32:
            raise ValueError("rank-four FP32 Q/K/V required")
        if t.layout != torch.strided or not t.is_cuda or not t.is_contiguous() or t.is_neg() or t.is_conj() or t.requires_grad:
            raise ValueError("contiguous materialized CUDA inference Q/K/V required")
        if t.shape != q.shape or t.device != q.device:
            raise ValueError("Q/K/V shape and device mismatch")
        if any(n < 0 or n >= 2**31 for n in t.shape):
            raise ValueError("signed32 Q/K/V shape required")
    batch, heads, length, dim = q.shape
    if heads <= 0 or dim not in (16, 32, 64, 128):
        raise ValueError("positive heads and supported feature dimension required")
    nr = triton.cdiv(length, bm)
    metas = (seqlens, block_count, block_offset, column_count, column_index)
    for t in metas:
        if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or not t.is_cuda or t.device != q.device:
            raise ValueError("int32 metadata on the Q device required")
        if t.layout != torch.strided or not t.is_contiguous() or t.is_neg() or t.is_conj():
            raise ValueError("contiguous materialized integer metadata required")
        if any(n < 0 or n >= 2**31 for n in t.shape):
            raise ValueError("signed32 metadata shape required")
    if seqlens.shape != (batch,) or block_count.shape != (batch, heads, nr) or column_count.shape != (batch, heads, nr):
        raise ValueError("sequence/count shape mismatch")
    if block_offset.ndim != 4 or block_offset.shape[:3] != (batch, heads, nr):
        raise ValueError("block offset shape mismatch")
    if column_index.ndim != 4 or column_index.shape[:3] != (batch, heads, nr):
        raise ValueError("column index shape mismatch")
    ns, nv = block_offset.shape[-1], column_index.shape[-1]
    if batch == 0 or length == 0:
        return False, 0
    if batch * heads > 65535 or nr >= 2**31:
        raise ValueError("CUDA grid axis bound")
    dense = (batch * heads * length + length + bm + bn) * dim + dim + 1
    metadata = (batch * heads * nr) * (ns + nv + 1) + ns + nv + bn + 1
    scalars = batch + heads + length + nr + ns + nv + bm + bn + dim + 1
    envelope = dense + metadata + scalars
    if envelope >= 2**31:
        raise ValueError("formed signed32 integer envelope")
    strides = (heads * length * dim, length * dim, dim, 1)
    for t in (q, k, v):
        if t.stride() != strides:
            raise ValueError("selected canonical dense strides required")
    for name, t in zip(("Q", "K", "V", "seqlens", "block_count", "block_offset", "column_count", "column_index"), (q, k, v, *metas)):
        _storage(t, envelope, name)
    lengths = seqlens.detach().cpu().tolist()
    blocks = block_count.detach().reshape(batch * heads, nr).cpu().tolist()
    counts = column_count.detach().reshape(batch * heads, nr).cpu().tolist()
    offsets = block_offset.detach().reshape(batch * heads, nr, ns).cpu().tolist()
    columns = column_index.detach().reshape(batch * heads, nr, nv).cpu().tolist()
    if any(n < 0 or n > length for n in lengths):
        raise ValueError("sequence length outside allocated context")
    for hz in range(batch * heads):
        seq = lengths[hz // heads]
        for row in range(nr):
            nb, nc = blocks[hz][row], counts[hz][row]
            if nb < 0 or nb > ns or nc < 0 or nc > nv:
                raise ValueError("sparse count outside its allocated capacity")
            if any(n < 0 or n >= seq for n in offsets[hz][row][:nb]):
                raise ValueError("selected block start outside sequence")
            if any(n < 0 or n >= seq for n in columns[hz][row][:nc]):
                raise ValueError("selected explicit column outside sequence")
    return True, envelope


def _output_storage(out, inputs, envelope):
    _storage(out, envelope, "output")
    lo, hi = out.data_ptr(), out.data_ptr() + 4 * out.numel()
    for t in inputs:
        base, size = t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()
        if size and not (hi <= base or base + size <= lo):
            raise ValueError("fresh output overlaps input backing")


@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    # Traverse every count-selected block.
    max_num_blks = num_blks
    for block_index in range(max_num_blks):
        # Use a conditional mask instead of break.
        cond = block_index < num_blks
        # Perform the following operations only when cond is True.
        start_n = tl.load(blks_ptr + block_index, mask=cond, other=0)
        cols = start_n + offs_n
        n_mask = (cols < seqlen) & cond
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = (cols[None, :] <= offs_m[:, None])
        qk = tl.where(m_mask & n_mask[None, :] & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Traverse every count-selected column.
    max_num_cols = num_cols
    for start_n in range(0, max_num_cols, BLOCK_N):
        # Use a conditional mask instead of break.
        cond = start_n < num_cols
        n_mask = (start_n + offs_n < num_cols) & cond
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = tl.where(l_i[:, None] == 0, 0.0, acc / l_i[:, None])
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


@triton.jit
def _triton_mixed_sparse_attn_fused_mask_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    # Traverse every count-selected block.
    max_num_blks = num_blks
    for block_index in range(max_num_blks):
        # Use a conditional mask instead of break.
        cond = block_index < num_blks
        # Perform the following operations only when cond is True.
        start_n = tl.load(blks_ptr + block_index, mask=cond, other=0)
        cols = start_n + offs_n
        n_mask = (cols < seqlen) & cond
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = (cols[None, :] <= offs_m[:, None])
        qk = tl.where(m_mask & n_mask[None, :] & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Traverse every count-selected column.
    max_num_cols = num_cols
    for start_n in range(0, max_num_cols, BLOCK_N):
        # Use a conditional mask instead of break.
        cond = start_n < num_cols
        n_mask = (start_n + offs_n < num_cols) & cond
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.where(m_mask & n_mask, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = tl.where(l_i[:, None] == 0, 0.0, acc / l_i[:, None])
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_mixed_sparse_attention(
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens: torch.Tensor,    # [BATCH, ]
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    launch, envelope = _validate(q, k, v, seqlens, block_count, block_offset,
                                  column_count, column_index, sm_scale,
                                  block_size_M, block_size_N)
    Lk = q.shape[-1]
    o = torch.zeros_like(q)
    if not launch:
        return o
    _output_storage(o, (q, k, v, seqlens, block_count, block_offset, column_count, column_index), envelope)
    sm_scale = float(sm_scale)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.float16
    with torch.cuda.device(q.device):
        _triton_mixed_sparse_attn_fwd_kernel[grid](
            q, k, v, seqlens, sm_scale,
            block_count, block_offset, column_count, column_index,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
            BLOCK_M=block_size_M, BLOCK_N=block_size_N,
            BLOCK_DMODEL=Lk,
            dtype=dtype,
            num_warps=4, num_stages=2,
        )

    return o

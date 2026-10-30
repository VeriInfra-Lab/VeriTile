
import torch
import triton
import triton.language as tl

# Triton kernel for the inner loop of attention forward pass
@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, 
                    K_ptrs, K_scale_ptr, V_ptrs,  
                    start_m,  
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    N_CTX: tl.constexpr, stride_kn, stride_vk):
    lo, hi = 0, N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & ((tl.arange(0, 128) < 96)[:, None])
        k = tl.load(K_ptrs, mask=k_mask, other=0.0)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        qk = tl.where(offs_n[None, :] < N_CTX - start_n, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < N_CTX - start_n) & ((tl.arange(0, 128) < 96)[None, :]), other=0.0)
        v = v.to(tl.float16)
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)  
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vk
    return acc, l_i

# Triton kernel for the attention forward pass
@triton.jit
def _attn_fwd_shifted(Q, K, V, Q_scale, K_scale, Out,  
              stride_qz, stride_qh, stride_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              Z, H, N_CTX,  
              HEAD_DIM: tl.constexpr,  
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr  
              ):
    tl.static_assert(BLOCK_M == 128)
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N) 
    

    offs_m = (start_m << 7) + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, 128).to(tl.int64)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + k_offset + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    O_block_ptr = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < 96)[None, :]), other=0.0)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,  
                                    start_m,  
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, N_CTX, stride_kn, stride_vk 
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < 96)[None, :]))

# Production API: raw, resolved, canonical contiguous FP32 inputs only.
# The explicit FP16 intermediates and BF16 output above are part of the source.
_INT32_MAX = 2**31 - 1
_UINT64_LIMIT = 2**64


def _launch_envelope(batch, heads, length, strides):
    qb, kb = triton.cdiv(length, 128), triton.cdiv(length, 64)
    extent = (batch + heads + length + 256 + batch * heads
              + (batch * heads + 1) * (qb + kb + 2))
    for s0, s1, s2, s3 in strides:
        extent += batch * s0 + heads * s1 + (length + 128) * s2 + 128 * s3
    if extent > _INT32_MAX or batch * heads > 65535:
        raise ValueError("shape/stride envelope exceeds source or launch integer bounds")
    return extent


def _check_pointer(tensor, extent):
    base, width = tensor.data_ptr(), tensor.element_size()
    if base < 0 or base % width or base + width * extent >= _UINT64_LIMIT:
        raise ValueError("formed address range must be aligned and fit unsigned64")


def forward_shifted(q, k, v, q_scale, k_scale):
    tensors = (q, k, v, q_scale, k_scale)
    if any(type(t) is not torch.Tensor for t in tensors):
        raise TypeError("ordinary torch.Tensor arguments required")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("ordinary strided input tensors required")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("FP32 Q, K, V and scales required")
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("all inputs must be on one CUDA device")
    if any(t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("resolved input tensors required")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("matching [Z,H,N,96] Q, K and V required")
    batch, heads, length, dim = q.shape
    if dim != 96:
        raise ValueError("logical D must be 96; compute padding remains 128")
    qb, kb = triton.cdiv(length, 128), triton.cdiv(length, 64)
    if q_scale.shape != (batch, heads, qb) or k_scale.shape != (batch, heads, kb):
        raise ValueError("packed scales must be [Z,H,ceil(N/128)] and [Z,H,ceil(N/64)]")
    dense = (heads * length * 96, length * 96, 96, 1)
    for tensor in (q, k, v):
        if tensor.numel() and tensor.stride() != dense:
            raise ValueError("canonical contiguous Q/K/V required")
    for tensor, blocks in ((q_scale, qb), (k_scale, kb)):
        if tensor.numel() and tensor.stride() != (heads * blocks, blocks, 1):
            raise ValueError("canonical packed scales required")
    o = torch.empty_like(q, dtype=torch.bfloat16)
    # The source allocates M but never reads or writes its cells.
    M = torch.empty((batch, heads, length), device=q.device, dtype=torch.float32)
    if batch == 0 or heads == 0 or length == 0:
        return o
    strides = (q.stride(), k.stride(), v.stride(), o.stride())
    extent = _launch_envelope(batch, heads, length, strides)
    for tensor in (*tensors, o):
        _check_pointer(tensor, extent)
    with torch.cuda.device(q.device):
        _attn_fwd_shifted[(qb, batch * heads, 1)](
            q, k, v, q_scale, k_scale, o,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(),
            batch, heads, N_CTX=length, HEAD_DIM=96, BLOCK_M=128, BLOCK_N=64,
            STAGE=1, num_warps=8, num_stages=2)
    return o

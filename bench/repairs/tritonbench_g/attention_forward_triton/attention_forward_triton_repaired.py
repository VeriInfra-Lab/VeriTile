
import triton
import triton.language as tl
import torch

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
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < N_CTX - start_n) & ((tl.arange(0, 128) < 96)[None, :]), other=0.0).to(tl.float16)
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v, out_dtype=tl.float16)  
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vk
    return acc, l_i

@triton.jit
def _attn_fwd(Q, K, V, Q_scale, K_scale, Out,  
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
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    v_offset = off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
    o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N) 
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
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

# Exact saved JIT pointer/integer specialization classes (SM80, 8 warps, 2 stages).
_SAVED_CLASSES = (
    ((16,16,16,16,16,16), (16,16,16,1,16,16,16,1,16,16,16,1,16,16,16,1,1,1,0)),
    ((16,16,0,16,16,16), (16,16,16,0,16,16,16,1,16,16,16,0,16,16,16,1,0,0,0)),
    ((16,0,16,16,16,16), (16,16,16,1,16,16,16,0,16,16,1,0,16,16,16,1,0,0,0)),
    ((0,0,0,16,16,16), (16,16,16,1,16,16,16,1,16,16,16,1,16,16,16,1,0,0,0)),
    ((16,16,16,16,16,16), (16,16,16,1,16,16,16,1,16,16,16,1,16,16,16,1,0,0,16)),
)


def _check_pointers(tensors, envelope):
    for t in tensors:
        base = t.data_ptr()
        if base < 0 or base % t.element_size() or base + 4 * envelope >= 2**64:
            raise ValueError("aligned formed addresses must fit unsigned64")


def _check_specialization(tensors, strides, shape):
    pointers = tuple(16 if t.data_ptr() % 16 == 0 else 0 for t in tensors)
    integers = tuple(1 if x == 1 else 16 if x % 16 == 0 else 0
                     for x in (*[x for stride in strides for x in stride], *shape))
    if (pointers, integers) not in _SAVED_CLASSES:
        raise ValueError("launch metadata must match a supported saved JIT specialization")


def forward(q, k, v, q_scale, k_scale):
    tensors = (q, k, v, q_scale, k_scale)
    if any(type(t) is not torch.Tensor for t in tensors):
        raise TypeError("ordinary torch.Tensor arguments required")
    if any(t.layout != torch.strided or not t.is_cuda for t in tensors):
        raise ValueError("ordinary strided CUDA tensors required")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("FP32 Q, K, V and scales required")
    if any(t.device != q.device or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("ordinary tensors on one device required")
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError("matching [B,H,N,96] Q, K and V required")
    batch, heads, length, dim = q.shape
    if dim != 96:
        raise ValueError("the logical head width is 96; compute padding is 128")
    qb, kb = triton.cdiv(length, 128), triton.cdiv(length, 64)
    if q_scale.shape != (batch, heads, qb) or k_scale.shape != (batch, heads, kb):
        raise ValueError("scales must be [B,H,ceil(N/128)] and [B,H,ceil(N/64)]")
    for t in (q, k, v):
        if t.numel() and any(s <= 0 for s in t.stride()):
            raise ValueError("positive ordinary affine Q/K/V strides required")
    for t, blocks in ((q_scale, qb), (k_scale, kb)):
        if t.numel() and t.stride() != (heads * blocks, blocks, 1):
            raise ValueError("canonical contiguous scales required")
    live_storage = [t.untyped_storage().data_ptr() for t in tensors if t.numel()]
    if len(set(live_storage)) != len(live_storage):
        raise ValueError("input allocations must be distinct")
    o = torch.empty_like(q, dtype=torch.float16)
    # Bound every formed offset, including masked lanes and final pointer advances.
    envelope = (batch + heads + length + 256 + batch * heads
                + (batch * heads + 1) * (qb + kb + 2))
    for t in (q, k, v, o):
        s0, s1, s2, s3 = t.stride()
        envelope += batch * s0 + heads * s1 + (length + 128) * s2 + 128 * s3
    if envelope >= 2**31 or batch * heads > 65535:
        raise ValueError("shape/stride envelope exceeds source or launch integer bounds")
    _check_pointers((*tensors, o), envelope)
    M = torch.empty((batch, heads, length), device=q.device, dtype=torch.float32)
    if batch == 0 or heads == 0 or length == 0:
        return o
    _check_specialization((*tensors, o), [t.stride() for t in (q,k,v,o)],
                          (batch,heads,length))
    BLOCK_M = 128
    BLOCK_N = 64
    HEAD_DIM_K = dim
    stage = 1
    grid = (qb, batch * heads, 1)
    with torch.cuda.device(q.device):
        _attn_fwd[grid](
            q, k, v, q_scale, k_scale, o,  
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  
            q.shape[0], q.shape[1],  
            N_CTX=q.shape[2],  
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,  
            STAGE=stage,  
            num_warps=8,  
            num_stages=2)
    return o





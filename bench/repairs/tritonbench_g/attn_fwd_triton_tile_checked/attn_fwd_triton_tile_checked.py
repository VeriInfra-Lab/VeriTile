# Separate repair candidate; original SHA256: de0fb466adb7844b34f53e44225c25411faad342e4192e8f9d5bbe71f154df12

import triton
import triton.language as tl
import torch

@triton.jit
def _attn_fwd_inner_repaired(acc, l_i, m_i, q, q_scale, 
                    K_ptrs, K_scale_ptr, V_ptrs,  
                    start_m, stride_kn, stride_vk,
                    BLOCK_DMODEL: tl.constexpr,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  
                    N_CTX: tl.constexpr):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += stride_kn * lo
        V_ptrs += stride_vk * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & ((tl.arange(0, BLOCK_DMODEL) < HEAD_DIM)[:, None])
        k = tl.load(K_ptrs, mask=k_mask, other=0)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask & (offs_n[None, :] < (N_CTX - start_n)), qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            qk = tl.where(offs_n[None, :] < (N_CTX - start_n), qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < (N_CTX - start_n)) & ((tl.arange(0, BLOCK_DMODEL) < HEAD_DIM)[None, :]), other=0)
        p = p.to(tl.float16)
        
        acc += tl.dot(p, v.to(tl.float16), out_dtype=tl.float16)  
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vk
    return acc, l_i, m_i

@triton.jit
def _attn_fwd_repaired(Q, K, V, Q_scale, K_scale, Out,  
              stride_qz, stride_qh, stride_qm, stride_qk,  
              stride_kz, stride_kh, stride_kn, stride_kk,  
              stride_vz, stride_vh, stride_vk, stride_vn,  
              stride_oz, stride_oh, stride_om, stride_on,  
              Z, H, N_CTX,  
              HEAD_DIM: tl.constexpr,
              BLOCK_DMODEL: tl.constexpr,
              BLOCK_M: tl.constexpr,  
              BLOCK_N: tl.constexpr,  
              STAGE: tl.constexpr):
    n_ctx = tl.cast(N_CTX, tl.int64)
    stride_kn64 = tl.cast(stride_kn, tl.int64)
    stride_vk64 = tl.cast(stride_vk, tl.int64)
    start_m = tl.program_id(0).to(tl.int64)
    off_hz = tl.program_id(1).to(tl.int64)
    
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    q_scale_offset = off_hz * tl.cdiv(n_ctx, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(n_ctx, BLOCK_N)
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + k_offset + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    O_block_ptr = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < n_ctx) & ((tl.arange(0, BLOCK_DMODEL) < HEAD_DIM)[None, :]), other=0)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner_repaired(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,  
                                    start_m, stride_kn64, stride_vk64, BLOCK_DMODEL,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,  
                                    4 - STAGE, offs_m, offs_n, n_ctx)
    acc, l_i, _ = _attn_fwd_inner_repaired(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,  
                                  start_m, stride_kn64, stride_vk64, BLOCK_DMODEL,
                                  BLOCK_M, HEAD_DIM, BLOCK_N,  
                                  2, offs_m, offs_n, n_ctx)
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask=(offs_m[:, None] < n_ctx) & ((tl.arange(0, BLOCK_DMODEL) < HEAD_DIM)[None, :]))

def _input_contract(q, k, v, q_scale, k_scale):
    tensors = (q, k, v, q_scale, k_scale)
    if any(type(t) is not torch.Tensor for t in tensors):
        raise TypeError("ordinary Tensor inputs required")
    if any(t.dtype != torch.float32 or t.layout != torch.strided or not t.is_cuda
           or t.device != q.device or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("resolved FP32 strided CUDA tensors on one device required")
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("matching rank-four Q/K/V required")
    if any(any(s < 0 for s in t.stride()) for t in (q, k, v)):
        raise ValueError("nonnegative source strides required")


def _launch_contract(q, k, v, q_scale, k_scale, o):
    batch, heads, rows, dim = q.shape
    bd = max(16, triton.next_power_of_2(dim))
    qm, kn = triton.cdiv(rows, 128), triton.cdiv(rows, 64)
    if not q_scale.is_contiguous() or not k_scale.is_contiguous():
        raise ValueError("scales must already have the source packed-flat layout")
    if q_scale.numel() < batch*heads*qm or k_scale.numel() < batch*heads*kn:
        raise ValueError("packed scales do not cover all source blocks")
    if qm > 2**31-1 or batch*heads > 65535 or bd > 1048576:
        raise ValueError("source launch axes and tile dimensions exceed the backend ABI")
    if 128 * bd > 1048576:
        raise ValueError("source tile exceeds Triton static tensor element limit")
    envelope = batch + heads + rows + dim + bd + 512 + batch*heads
    for t in (q, k, v, o):
        sz, sh, sr, sd = t.stride()
        envelope += batch*sz + heads*sh + (rows+256)*sr + bd*sd + 1
    envelope += batch*heads*qm + qm + 2 + batch*heads*kn + kn + 2
    if envelope >= 2**63:
        raise ValueError("all padded and advanced source offsets must fit signed64")
    for t in (q, k, v, q_scale, k_scale, o):
        base, width = t.data_ptr(), t.element_size()
        if base < 0 or base % width or base + width*envelope >= 2**64:
            raise ValueError("naturally aligned formed addresses must fit unsigned64")
    return envelope


def forward_repaired(q, k, v, q_scale, k_scale):
    _input_contract(q, k, v, q_scale, k_scale)
    BLOCK_M = 128
    BLOCK_N = 64
    assert q.ndim == k.ndim == v.ndim == 4
    assert q.shape == k.shape == v.shape
    assert q.device == k.device == v.device == q_scale.device == k_scale.device
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    o = torch.empty_like(q, dtype=torch.bfloat16)
    # Preserve the original unused workspace allocation and its API outcome.
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    if q.numel() == 0:
        return o
    assert q_scale.numel() >= q.shape[0] * q.shape[1] * triton.cdiv(q.shape[2], BLOCK_M)
    assert k_scale.numel() >= q.shape[0] * q.shape[1] * triton.cdiv(q.shape[2], BLOCK_N)
    _launch_contract(q, k, v, q_scale, k_scale, o)
    stage = 3

    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    with torch.cuda.device(q.device):
        _attn_fwd_repaired[grid](
            q, k, v, q_scale, k_scale, o,  
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  
            q.shape[0], q.shape[1],  
            N_CTX=q.shape[2],  
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=HEAD_DIM_K,
            BLOCK_DMODEL=max(16, triton.next_power_of_2(HEAD_DIM_K)),  
            STAGE=stage,  
            num_warps=8,  
            num_stages=3)
    return o

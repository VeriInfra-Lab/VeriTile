import VeriTile.Triton.DSL.LoopJitFunctions

namespace VeriTile.Bench.Repairs.AttentionForwardTriton
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false

def signature : CallSignature where
  pointers := [("Q",.fp32),("K",.fp32),("V",.fp32),("Q_scale",.fp32),("K_scale",.fp32),("Out",.fp16)]
  scalars := [("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("Z",.integer),("H",.integer),("N_CTX",.integer)]
  constexpr := ["HEAD_DIM","BLOCK_M","BLOCK_N","STAGE"]

def originalSource : StatementJitFunction := with_triton_params (acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX, stride_kn, stride_vk) constexpr_args [BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX] in triton {
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
}

def originalKernel (Q K V Q_scale K_scale Out : RegionName) (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N STAGE : Nat) : ComputeKernel :=
  with_triton_call signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, Z := Z, H := H, N_CTX := N_CTX]
    jit_functions [_attn_fwd_inner := originalSource] in triton {
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
}

def candidateSource : StatementJitFunction := with_triton_params (acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX, stride_kn, stride_vk, head_mask) constexpr_args [BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX] in triton {
    lo, hi = 0, N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & ((head_mask)[:, None])
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
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < N_CTX - start_n) & ((head_mask)[None, :]), other=0.0).to(tl.float16)
        p = p.to(tl.float16)

        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * stride_vk
    return acc, l_i
}

def candidateKernel (Q K V Q_scale K_scale Out : RegionName) (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N STAGE : Nat) : ComputeKernel :=
  with_triton_call signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, Z := Z, H := H, N_CTX := N_CTX]
    jit_functions [_attn_fwd_inner_head_mask := candidateSource] in triton {
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
    head_mask = offs_k < 96
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + k_offset + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    O_block_ptr = Out + o_offset + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((head_mask)[None, :]), other=0.0)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i = _attn_fwd_inner_head_mask(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
                                    start_m,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4 - STAGE, offs_m, offs_n, N_CTX, stride_kn, stride_vk, head_mask
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < N_CTX) & ((head_mask)[None, :]))
}

end
end VeriTile.Bench.Repairs.AttentionForwardTriton

import VeriTile.Triton.DSL.SpecializedJitFunctions
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttnFwdCausal.TypedSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 24000
set_option maxHeartbeats 3000000

def innerSource : SpecializedStatementJitFunction :=
  with_triton_params (acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs, start_m, BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX) constexpr_args [BLOCK_M, HEAD_DIM, BLOCK_N, STAGE, offs_m, offs_n, N_CTX] in triton {
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
        lo = tl.multiple_of(lo, BLOCK_M)
        K_scale_ptr += lo // BLOCK_N
        K_ptrs += HEAD_DIM * lo
        V_ptrs += HEAD_DIM * lo
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = (offs_n[None, :] < (N_CTX - start_n)) & ((tl.arange(0, 128) < HEAD_DIM)[:, None])
        k = tl.load(K_ptrs, mask=k_mask, other=0.0)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        mask = (offs_m[:, None] >= start_n + offs_n[None, :]) & (offs_n[None, :] < N_CTX - start_n)
        qk = tl.where(mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(V_ptrs, mask=(offs_n[:, None] < N_CTX - start_n) & ((tl.arange(0, 128) < HEAD_DIM)[None, :]), other=0.0).to(tl.float16)
        p = p.to(tl.float16)
        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * HEAD_DIM
        K_scale_ptr += 1
        V_ptrs += BLOCK_N * HEAD_DIM
    return acc, l_i, m_i
}

def signature : CallSignature where
  pointers := [("Q",.fp32),("K",.fp32),("V",.fp32),("Q_scale",.fp32),("K_scale",.fp32),("Out",.fp16)]
  scalars := [("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("Z",.integer),("H",.integer),("N_CTX",.integer)]
  constexpr := ["HEAD_DIM", "BLOCK_M", "BLOCK_N", "STAGE"]
  readPorts := ["Q","Q_scale","K","K_scale","V"]
  writePorts := ["Out"]

def repairedKernel (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  let STAGE : Nat:=3
  with_triton_call signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, Z := Z, H := H, N_CTX := N_CTX]
    jit_functions [_attn_fwd_inner := innerSource] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    vk_offset = qvk_offset // stride_qm
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N)


    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < HEAD_DIM)[None, :]), other=0.0)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
                                    start_m,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4 - STAGE, offs_m, offs_n, N_CTX
                                    )
    acc, l_i, _ = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
                                    start_m,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    2, offs_m, offs_n, N_CTX
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < HEAD_DIM)[None, :]))
}

def quotientRemainderKernel (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  let STAGE : Nat:=3
  with_triton_call signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, Z := Z, H := H, N_CTX := N_CTX]
    jit_functions [_attn_fwd_inner := innerSource] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz - (off_hz // H) * H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    vk_offset = qvk_offset // stride_qm
    q_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_M)
    k_scale_offset = off_hz * tl.cdiv(N_CTX, BLOCK_N)


    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, 128)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + qvk_offset + offs_k[:, None] + offs_n[None, :] * stride_kn
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + qvk_offset + offs_n[:, None] * stride_qm + offs_k[None, :] * stride_qk
    O_block_ptr = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    q = tl.load(Q_ptrs, mask=(offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < HEAD_DIM)[None, :]), other=0.0)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
                                    start_m,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4 - STAGE, offs_m, offs_n, N_CTX
                                    )
    acc, l_i, _ = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, K_ptrs, K_scale_ptr, V_ptrs,
                                    start_m,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    2, offs_m, offs_n, N_CTX
                                    )
    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), mask = (offs_m[:, None] < N_CTX) & ((tl.arange(0, 128) < HEAD_DIM)[None, :]))
}

end
end VeriTile.Bench.Repairs.AttnFwdCausal.TypedSource

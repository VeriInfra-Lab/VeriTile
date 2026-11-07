import bench.tritonbench_g.attention_kernel_aligned.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.ScalarInfinityWhere
import VeriTile.Meta.StatementAudit

/-! Complete repaired FP16/BF16 bodies and existing early-query candidates.
Storage, scalar bindings and output type metadata are outside the quotations. -/
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Pasted
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 8000000

def fp16Signature : CallSignature where
  pointers := [("Q",.fp16),("K",.fp16),("V",.fp16),("B0",.fp16),("Out",.fp16)]
  scalars := [("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("stride_b0z",.integer),("stride_b0h",.integer),("stride_b0m",.integer),("Z",.integer),("H",.integer),("N_CTX",.integer),("P_SEQ",.integer),("sm_scale",.fp32)]
  constexpr := ["OUT_DTYPE","BIAS_LAST_SIZE","B0_NUMEL","BLOCK_DMODEL","BLOCK_M","BLOCK_N"]
  readPorts := ["Q","K","V","B0"]
  writePorts := ["Out"]

def bf16Signature : CallSignature where
  pointers := [("Q",.bf16),("K",.bf16),("V",.bf16),("B0",.bf16),("Out",.bf16)]
  scalars := [("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("stride_b0z",.integer),("stride_b0h",.integer),("stride_b0m",.integer),("Z",.integer),("H",.integer),("N_CTX",.integer),("P_SEQ",.integer),("sm_scale",.fp32)]
  constexpr := ["OUT_DTYPE","BIAS_LAST_SIZE","B0_NUMEL","BLOCK_DMODEL","BLOCK_M","BLOCK_N"]
  readPorts := ["Q","K","V","B0"]
  writePorts := ["Out"]

def originalHalfKernel (Q K V B0 Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ : Nat) (sm_scale : ℝ)
    (BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  let «OUT_DTYPE» : FloatDType := .fp16
  with_triton_call fp16Signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, stride_b0z := stride_b0z, stride_b0h := stride_b0h, stride_b0m := stride_b0m, Z := Z, H := H, N_CTX := N_CTX, P_SEQ := P_SEQ, sm_scale := sm_scale] scalar_infinity_where in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_b * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k, out_dtype=OUT_DTYPE)

        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
        qk += (b0 + b1)
        qk = tl.where((start_n + tl.arange(0, BLOCK_N))[None, :] < N_CTX, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0, 1))
  }

def originalBF16Kernel (Q K V B0 Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ : Nat) (sm_scale : ℝ)
    (BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  let «OUT_DTYPE» : FloatDType := .bf16
  with_triton_call bf16Signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, stride_b0z := stride_b0z, stride_b0h := stride_b0h, stride_b0m := stride_b0m, Z := Z, H := H, N_CTX := N_CTX, P_SEQ := P_SEQ, sm_scale := sm_scale] scalar_infinity_where in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_b * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k, out_dtype=tl.float32).to(OUT_DTYPE)

        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
        qk += (b0 + b1)
        qk = tl.where((start_n + tl.arange(0, BLOCK_N))[None, :] < N_CTX, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0, 1))
  }

def earlyHalfKernel (Q K V B0 Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ : Nat) (sm_scale : ℝ)
    (BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  let «OUT_DTYPE» : FloatDType := .fp16
  with_triton_call fp16Signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, stride_b0z := stride_b0z, stride_b0h := stride_b0h, stride_b0m := stride_b0m, Z := Z, H := H, N_CTX := N_CTX, P_SEQ := P_SEQ, sm_scale := sm_scale] scalar_infinity_where in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_b * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k, out_dtype=OUT_DTYPE)

        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
        qk += (b0 + b1)
        qk = tl.where((start_n + tl.arange(0, BLOCK_N))[None, :] < N_CTX, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0, 1))
  }

def earlyBF16Kernel (Q K V B0 Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ : Nat) (sm_scale : ℝ)
    (BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  let «OUT_DTYPE» : FloatDType := .bf16
  with_triton_call bf16Signature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, stride_b0z := stride_b0z, stride_b0h := stride_b0h, stride_b0m := stride_b0m, Z := Z, H := H, N_CTX := N_CTX, P_SEQ := P_SEQ, sm_scale := sm_scale] scalar_infinity_where in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_b = off_hz // H
    off_h = off_hz % H
    q_offset = off_b * stride_qz + off_h * stride_qh
    k_offset = off_b * stride_kz + off_h * stride_kh
    v_offset = off_b * stride_vz + off_h * stride_vh
    o_offset = off_b * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    qk_scale = sm_scale * 1.44269504
    q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, N_CTX + P_SEQ),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_CTX + P_SEQ, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M)

    b_offset = off_b * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
    for start_n in range(lo, hi, BLOCK_N):
        k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k, out_dtype=tl.float32).to(OUT_DTYPE)

        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0)
        qk += (b0 + b1)
        qk = tl.where((start_n + tl.arange(0, BLOCK_N))[None, :] < N_CTX, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0, 1))
  }

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Pasted

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
def typedSurface (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionKernelAligned.Pasted.originalHalfKernel Q K V B0 Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ sm_scale BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N

def typedSurfaceQueryEarly (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionKernelAligned.Pasted.earlyHalfKernel Q K V B0 Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ sm_scale BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N

def source (p : Parameters) : ComputeKernel :=
  typedSurface p.q p.k p.v p.bias p.output p.scale
    p.qBatchStride p.qHeadStride p.qRowStride p.qColStride
    p.kBatchStride p.kHeadStride p.kSeqStride p.kDimStride
    p.vBatchStride p.vHeadStride p.vSeqStride p.vDimStride
    p.outBatchStride p.outHeadStride p.outRowStride p.outColStride
    p.biasBatchStride p.biasHeadStride p.biasRowStride p.batches p.heads
    p.length 0 p.biasLast p.biasNumel p.head p.blockM p.blockN

def queryEarly (p : Parameters) : ComputeKernel :=
  typedSurfaceQueryEarly p.q p.k p.v p.bias p.output p.scale
    p.qBatchStride p.qHeadStride p.qRowStride p.qColStride
    p.kBatchStride p.kHeadStride p.kSeqStride p.kDimStride
    p.vBatchStride p.vHeadStride p.vSeqStride p.vDimStride
    p.outBatchStride p.outHeadStride p.outRowStride p.outColStride
    p.biasBatchStride p.biasHeadStride p.biasRowStride p.batches p.heads
    p.length 0 p.biasLast p.biasNumel p.head p.blockM p.blockN


end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Source

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
def typedSurface (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionKernelAligned.Pasted.originalBF16Kernel Q K V B0 Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ sm_scale BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N

def typedSurfaceQueryEarly (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionKernelAligned.Pasted.earlyBF16Kernel Q K V B0 Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ sm_scale BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N

def source (p : Parameters) : ComputeKernel :=
  typedSurface p.q p.k p.v p.bias p.output p.scale
    p.qBatchStride p.qHeadStride p.qRowStride p.qColStride
    p.kBatchStride p.kHeadStride p.kSeqStride p.kDimStride
    p.vBatchStride p.vHeadStride p.vSeqStride p.vDimStride
    p.outBatchStride p.outHeadStride p.outRowStride p.outColStride
    p.biasBatchStride p.biasHeadStride p.biasRowStride p.batches p.heads
    p.length 0 p.biasLast p.biasNumel p.head p.blockM p.blockN

def queryEarly (p : Parameters) : ComputeKernel :=
  typedSurfaceQueryEarly p.q p.k p.v p.bias p.output p.scale
    p.qBatchStride p.qHeadStride p.qRowStride p.qColStride
    p.kBatchStride p.kHeadStride p.kSeqStride p.kDimStride
    p.vBatchStride p.vHeadStride p.vSeqStride p.vDimStride
    p.outBatchStride p.outHeadStride p.outRowStride p.outColStride
    p.biasBatchStride p.biasHeadStride p.biasRowStride p.batches p.heads
    p.length 0 p.biasLast p.biasNumel p.head p.blockM p.blockN


end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source

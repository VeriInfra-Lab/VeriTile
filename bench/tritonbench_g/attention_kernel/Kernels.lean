import bench.tritonbench_g.attention_kernel.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.OutputDTypeMetadata
import VeriTile.Meta.StatementAudit

/-! Complete repaired Python attention body and the existing early-query
comparison. All pointer types and scalar/constexpr bindings are caller metadata. -/
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 6000000

def callSignature (storage : Storage) : CallSignature where
  pointers := [("Q", match storage with | .fp16 => .fp16 | .bf16 => .bf16),("K", match storage with | .fp16 => .fp16 | .bf16 => .bf16),("V", match storage with | .fp16 => .fp16 | .bf16 => .bf16),("B0", match storage with | .fp16 => .fp16 | .bf16 => .bf16),("Out", match storage with | .fp16 => .fp16 | .bf16 => .bf16)]
  scalars := [("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vk",.integer),("stride_vn",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_on",.integer),("stride_b0z",.integer),("stride_b0h",.integer),("stride_b0m",.integer),("Z",.integer),("H",.integer),("N_CTX",.integer),("P_SEQ",.integer),("sm_scale",.fp32)]
  constexpr := ["OUT_DTYPE","BIAS_LAST_SIZE","B0_NUMEL","BLOCK_DMODEL","BLOCK_M","BLOCK_N"]
  readPorts := ["Q","K","V","B0"]
  writePorts := ["Out"]

def originalKernel (storage : Storage) (Q K V B0 Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ : Nat) (sm_scale : ℝ)
    («OUT_DTYPE» : FloatDType) (BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature storage) call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, stride_b0z := stride_b0z, stride_b0h := stride_b0h, stride_b0m := stride_b0m, Z := Z, H := H, N_CTX := N_CTX, P_SEQ := P_SEQ, sm_scale := sm_scale] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
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

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
    q = (q * qk_scale).to(OUT_DTYPE)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M).to(tl.int64)

    b_offset = off_z * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0.0)
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        # , boundary_check=(0, 1), padding_option="zero")
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        # , boundary_check=(1, 0), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k) #, out_dtype=OUT_DTYPE)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias
        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0.0)
        qk += ((b0 + b1) * 1.44269504)
        qk = tl.where(tl.arange(0, BLOCK_N)[None, :] < N_CTX - start_n, qk, -float("inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0,))
  }

def queryEarlyKernel (storage : Storage) (Q K V B0 Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ : Nat) (sm_scale : ℝ)
    («OUT_DTYPE» : FloatDType) (BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature storage) call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_vz := stride_vz, stride_vh := stride_vh, stride_vk := stride_vk, stride_vn := stride_vn, stride_oz := stride_oz, stride_oh := stride_oh, stride_om := stride_om, stride_on := stride_on, stride_b0z := stride_b0z, stride_b0h := stride_b0h, stride_b0m := stride_b0m, Z := Z, H := H, N_CTX := N_CTX, P_SEQ := P_SEQ, sm_scale := sm_scale] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
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

    # initialize offsets
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX + P_SEQ

    b_ptr_offsets_m = tl.arange(0, BLOCK_M).to(tl.int64)

    b_offset = off_z * stride_b0z + off_h * stride_b0h
    b_ptr_offsets_n_1 = (tl.arange(0, BLOCK_N) %
                         BIAS_LAST_SIZE) + BIAS_LAST_SIZE
    b1 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                 * stride_b0m)[:, None] + b_ptr_offsets_n_1[None, :],
                 mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0.0)
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        # , boundary_check=(0, 1), padding_option="zero")
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        # , boundary_check=(1, 0), padding_option="zero")
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=OUT_DTYPE)
        qk += tl.dot(q, k) #, out_dtype=OUT_DTYPE)

        # -- compute rel_h[:, None] + rel_w[None, :] bias ---

        # Bias
        b0 = tl.load(B0 + b_offset + ((start_m * BLOCK_M + b_ptr_offsets_m)
                     * stride_b0m)[:, None] + start_n // BLOCK_N,
                     mask=(start_m * BLOCK_M + b_ptr_offsets_m)[:, None] < N_CTX, other=0.0)
        qk += ((b0 + b1) * 1.44269504)
        qk = tl.where(tl.arange(0, BLOCK_N)[None, :] < N_CTX - start_n, qk, -float("inf"))

        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(OUT_DTYPE), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # write back l and m
    acc = acc / l_i[:, None]

    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(OUT_DTYPE), boundary_check=(0,))
  }

def repaired (storage : Storage) (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  originalKernel storage q k v bias out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ sm_scale storage.float BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N

def queryEarly (storage : Storage) (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  queryEarlyKernel storage q k v bias out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on stride_b0z stride_b0h stride_b0m Z H N_CTX P_SEQ sm_scale storage.float BIAS_LAST_SIZE B0_NUMEL BLOCK_DMODEL BLOCK_M BLOCK_N

namespace FP16
abbrev repaired := VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource.repaired .fp16
abbrev queryEarly := VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource.queryEarly .fp16
end FP16

namespace BF16
abbrev repaired := VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource.repaired .bf16
abbrev queryEarly := VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource.queryEarly .bf16
end BF16

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource

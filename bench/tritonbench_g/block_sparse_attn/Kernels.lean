import VeriTile.Triton.DSL.PointerLiteralOffsets
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000

/-- The selected checked Python kernel has fixed B16 tiles and FP32 storage.
All scalar arguments, all four Q/K/V strides, both masks, and the channel
loop remain explicit. Integer metadata is read from the two INT32 regions. -/
structure Parameters where
  rowStride : Nat
  colStride : Nat
  layouts : Nat
  scale : ℝ
  qb : Nat
  qh : Nat
  qm : Nat
  qd : Nat
  kb : Nat
  kh : Nat
  kn : Nat
  kd : Nat
  vb : Nat
  vh : Nat
  vn : Nat
  vd : Nat
  ob : Nat
  oh : Nat
  om : Nat
  heads : Nat
  kvHeads : Nat
  length : Nat
  blocks : Nat
  evenM : Bool
  evenN : Bool

set_option linter.unusedVariables false

def callSignature : CallSignature where
  pointers := [("out",.fp32),("Q",.fp32),("K",.fp32),("V",.fp32),("layout_csr_row_indices",.i32),("layout_csr_col_indices",.i32)]
  scalars := [("layout_csr_row_stride_h",.integer),("layout_csr_col_stride_h",.integer),("num_layout",.integer),("stride_qb",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qd",.integer),("stride_kb",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kd",.integer),("stride_vb",.integer),("stride_vh",.integer),("stride_vn",.integer),("stride_vd",.integer),("stride_ob",.integer),("stride_oh",.integer),("stride_om",.integer),("num_heads",.integer),("num_kv_heads",.integer),("total_seq_len",.integer),("softmax_scale",.fp32)]
  constexpr := ["BLOCK_M","EVEN_M","BLOCK_N","EVEN_N","BLOCK_D","NUM_D_BLOCKS"]
  readPorts := ["Q","K","V","layout_csr_row_indices","layout_csr_col_indices"]
  writePorts := ["out"]

def originalKernel (out Q K V : RegionName) (layout_csr_row_indices layout_csr_col_indices : Region .int) (layout_csr_row_stride_h layout_csr_col_stride_h num_layout stride_qb stride_qh stride_qm stride_qd stride_kb stride_kh stride_kn stride_kd stride_vb stride_vh stride_vn stride_vd stride_ob stride_oh stride_om num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D NUM_D_BLOCKS : Nat) (EVEN_M EVEN_N : Bool) : ComputeKernel :=
  with_triton_call callSignature call_args [layout_csr_row_stride_h := layout_csr_row_stride_h, layout_csr_col_stride_h := layout_csr_col_stride_h, num_layout := num_layout, stride_qb := stride_qb, stride_qh := stride_qh, stride_qm := stride_qm, stride_qd := stride_qd, stride_kb := stride_kb, stride_kh := stride_kh, stride_kn := stride_kn, stride_kd := stride_kd, stride_vb := stride_vb, stride_vh := stride_vh, stride_vn := stride_vn, stride_vd := stride_vd, stride_ob := stride_ob, stride_oh := stride_oh, stride_om := stride_om, num_heads := num_heads, num_kv_heads := num_kv_heads, total_seq_len := total_seq_len] in triton {
    q_seq_len = total_seq_len

    # Grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads, NUM_D_BLOCKS]
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    channel_block = tl.program_id(2)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads

    # For group query attention, map the query head index to the corresponding one for key and value.
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups

    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h_kv * stride_kh
    V += off_b * stride_vb + off_h_kv * stride_vh

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd  # [BLOCK_M, BLOCK_D]
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd  # [BLOCK_D, BLOCK_N]
    off_v = offs_n[:, None] * stride_vn + (channel_block * BLOCK_D + offs_d[None, :]) * stride_vd  # [BLOCK_N, BLOCK_D]

    # Initialize pointers to query, key, value
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    layout_h = off_h % num_layout

    # This assumes that past sequence length is 0, otherwise need + (past_seq_len + 1) // BLOCK_M.
    layout_ptr = layout_csr_row_indices + layout_h * layout_csr_row_stride_h + start_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + 1).to(tl.int32)

    # Loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # Fold every channel block; the third grid axis owns one output block.
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for score_block in tl.static_range(NUM_D_BLOCKS):
            if EVEN_M:
                q = tl.load(q_ptrs + score_block * BLOCK_D * stride_qd)
            else:
                q = tl.load(q_ptrs + score_block * BLOCK_D * stride_qd,
                            mask=offs_m[:, None] < q_seq_len, other=0)
            if EVEN_N:
                k = tl.load(k_ptrs + start_n * stride_kn + score_block * BLOCK_D * stride_kd)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn + score_block * BLOCK_D * stride_kd,
                            mask=offs_n[None, :] + start_n < total_seq_len, other=0)
            qk += tl.dot(q, k)

        qk *= softmax_scale

        # This assumes that past sequence length is 0, otherwise need offs_m[:, None] + past_seq_len >= ...
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)
        # update acc
        if EVEN_N:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < total_seq_len, other=0)
        acc += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_o = off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + channel_block * BLOCK_D + offs_d[None, :]
    out_ptrs = out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
  }

def quotientRemainderKernel (out Q K V : RegionName) (layout_csr_row_indices layout_csr_col_indices : Region .int) (layout_csr_row_stride_h layout_csr_col_stride_h num_layout stride_qb stride_qh stride_qm stride_qd stride_kb stride_kh stride_kn stride_kd stride_vb stride_vh stride_vn stride_vd stride_ob stride_oh stride_om num_heads num_kv_heads total_seq_len BLOCK_M BLOCK_N BLOCK_D NUM_D_BLOCKS : Nat) (EVEN_M EVEN_N : Bool) : ComputeKernel :=
  with_triton_call callSignature call_args [layout_csr_row_stride_h := layout_csr_row_stride_h, layout_csr_col_stride_h := layout_csr_col_stride_h, num_layout := num_layout, stride_qb := stride_qb, stride_qh := stride_qh, stride_qm := stride_qm, stride_qd := stride_qd, stride_kb := stride_kb, stride_kh := stride_kh, stride_kn := stride_kn, stride_kd := stride_kd, stride_vb := stride_vb, stride_vh := stride_vh, stride_vn := stride_vn, stride_vd := stride_vd, stride_ob := stride_ob, stride_oh := stride_oh, stride_om := stride_om, num_heads := num_heads, num_kv_heads := num_kv_heads, total_seq_len := total_seq_len] in triton {
    q_seq_len = total_seq_len

    # Grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads, NUM_D_BLOCKS]
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    channel_block = tl.program_id(2)

    off_h = off_bh - (off_bh // num_heads) * num_heads
    off_b = off_bh // num_heads

    # For group query attention, map the query head index to the corresponding one for key and value.
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups

    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h_kv * stride_kh
    V += off_b * stride_vb + off_h_kv * stride_vh

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd  # [BLOCK_M, BLOCK_D]
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd  # [BLOCK_D, BLOCK_N]
    off_v = offs_n[:, None] * stride_vn + (channel_block * BLOCK_D + offs_d[None, :]) * stride_vd  # [BLOCK_N, BLOCK_D]

    # Initialize pointers to query, key, value
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    layout_h = off_h % num_layout

    # This assumes that past sequence length is 0, otherwise need + (past_seq_len + 1) // BLOCK_M.
    layout_ptr = layout_csr_row_indices + layout_h * layout_csr_row_stride_h + start_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + 1).to(tl.int32)

    # Loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # Fold every channel block; the third grid axis owns one output block.
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for score_block in tl.static_range(NUM_D_BLOCKS):
            if EVEN_M:
                q = tl.load(q_ptrs + score_block * BLOCK_D * stride_qd)
            else:
                q = tl.load(q_ptrs + score_block * BLOCK_D * stride_qd,
                            mask=offs_m[:, None] < q_seq_len, other=0)
            if EVEN_N:
                k = tl.load(k_ptrs + start_n * stride_kn + score_block * BLOCK_D * stride_kd)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn + score_block * BLOCK_D * stride_kd,
                            mask=offs_n[None, :] + start_n < total_seq_len, other=0)
            qk += tl.dot(q, k)

        qk *= softmax_scale

        # This assumes that past sequence length is 0, otherwise need offs_m[:, None] + past_seq_len >= ...
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)
        # update acc
        if EVEN_N:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < total_seq_len, other=0)
        acc += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    off_o = off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + channel_block * BLOCK_D + offs_d[None, :]
    out_ptrs = out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
  }

def declared (p : Parameters) : ComputeKernel :=
  originalKernel "out" "Q" "K" "V" "rows" "cols" p.rowStride p.colStride p.layouts p.qb p.qh p.qm p.qd p.kb p.kh p.kn p.kd p.vb p.vh p.vn p.vd p.ob p.oh p.om p.heads p.kvHeads p.length 16 16 16 p.blocks p.evenM p.evenN

/-- Complete repaired source; call metadata stays outside the pasted body.
The existing invocation ABI supplies the runtime FP32 softmax scale. -/
def source (p : Parameters) : ComputeKernel := declared p

def quotientRemainder (p : Parameters) : ComputeKernel :=
  quotientRemainderKernel "out" "Q" "K" "V" "rows" "cols" p.rowStride p.colStride p.layouts p.qb p.qh p.qm p.qd p.kb p.kh p.kn p.kd p.vb p.vh p.vn p.vd p.ob p.oh p.om p.heads p.kvHeads p.length 16 16 16 p.blocks p.evenM p.evenN

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

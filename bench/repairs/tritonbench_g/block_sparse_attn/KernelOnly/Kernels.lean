import VeriTile.Triton.DSL
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

/-- Complete readable quotation of SHA256 f103646a18aadb536696840dc3e8014565d65d6f247a9ea9a7acd27efbdaf3b7.
The wrapper's selected BM=BN=BD=16 and FP32 storage are specialized here.
The two source add(dot) expressions are intentionally retained. -/
def declared (p : Parameters) : ComputeKernel :=
  let out : RegionName := "out"
  let Q : RegionName := "Q"
  let K : RegionName := "K"
  let V : RegionName := "V"
  let layout_csr_row_indices : Region .int := "rows"
  let layout_csr_col_indices : Region .int := "cols"
  let EVEN_M := p.evenM
  let EVEN_N := p.evenN
  triton {
  tl.ports inputs=[Q,K,V,Region.cast layout_csr_row_indices,Region.cast layout_csr_col_indices], outputs=[out]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, out = tl.float32
  tl.region layout_csr_row_indices = tl.int32, layout_csr_col_indices = tl.int32
  tl.param softmax_scale : tl.float32
  q_seq_len = $(p.length)
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  channel_block = tl.program_id(2)
  off_h = off_bh % $(p.heads)
  off_b = off_bh // $(p.heads)
  head_groups = $(p.heads) // $(p.kvHeads)
  off_h_kv = off_h // head_groups
  Q += off_b * $(p.qb) + off_h * $(p.qh)
  K += off_b * $(p.kb) + off_h_kv * $(p.kh)
  V += off_b * $(p.vb) + off_h_kv * $(p.vh)
  offs_m = start_m * $(16) + tl.arange(0, 16)
  offs_n = tl.arange(0, 16)
  offs_d = tl.arange(0, 16)
  off_q = offs_m[:, None] * $(p.qm) + offs_d[None, :] * $(p.qd)
  off_k = offs_n[None, :] * $(p.kn) + offs_d[:, None] * $(p.kd)
  off_v = offs_n[:, None] * $(p.vn) + (channel_block * $(16) + offs_d[None, :]) * $(p.vd)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_i = tl.zeros([16], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([16], dtype=tl.float32)
  acc = tl.zeros([16,16], dtype=tl.float32)
  layout_h = off_h % $(p.layouts)
  layout_ptr = layout_csr_row_indices + layout_h * $(p.rowStride) + start_m
  start_l = (tl.load(layout_ptr)).to(tl.int32)
  end_l = (tl.load(layout_ptr + $(1))).to(tl.int32)
  for col_idx_idx in range(start_l, end_l) {
    col_idx = (tl.load(layout_csr_col_indices +
      layout_h * $(p.colStride) + col_idx_idx)).to(tl.int32)
    start_n = col_idx * $(16)
    qk = tl.zeros([16,16], dtype=tl.float32)
    for score_block in tl.static_range($(p.blocks)) {
      if EVEN_M {
        q = tl.load(q_ptrs + score_block * $(16) * $(p.qd))
      } else {
        q = tl.load(q_ptrs + score_block * $(16) * $(p.qd),
          mask=offs_m[:, None] < q_seq_len, other=0)
      }
      if EVEN_N {
        k = tl.load(k_ptrs + start_n * $(p.kn) + score_block * $(16) * $(p.kd))
      } else {
        k = tl.load(k_ptrs + start_n * $(p.kn) + score_block * $(16) * $(p.kd),
          mask=offs_n[None, :] + start_n < $(p.length), other=0)
      }
      qk += tl.dot(q, k)
    }
    qk *= softmax_scale
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    p = (p).to(tl.float32)
    if EVEN_N {
      v = tl.load(v_ptrs + start_n * $(p.vn))
    } else {
      v = tl.load(v_ptrs + start_n * $(p.vn),
        mask=offs_n[:, None] + start_n < $(p.length), other=0)
    }
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = off_b * $(p.ob) + off_h * $(p.oh) +
    offs_m[:, None] * $(p.om) + channel_block * $(16) + offs_d[None, :]
  out_ptrs = out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
}

/-- Complete direct source; runtime scale is bound by the invocation ABI. -/
def source (p : Parameters) : ComputeKernel := declared p

/-- Quotient CSE candidate. All numerical operations and metadata loops are
identical; the quotient is also computed by off_b. -/
def quotientRemainder (p : Parameters) : ComputeKernel :=
  let out : RegionName := "out"
  let Q : RegionName := "Q"
  let K : RegionName := "K"
  let V : RegionName := "V"
  let layout_csr_row_indices : Region .int := "rows"
  let layout_csr_col_indices : Region .int := "cols"
  let EVEN_M := p.evenM
  let EVEN_N := p.evenN
  triton {
  tl.ports inputs=[Q,K,V,Region.cast layout_csr_row_indices,Region.cast layout_csr_col_indices], outputs=[out]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, out = tl.float32
  tl.region layout_csr_row_indices = tl.int32, layout_csr_col_indices = tl.int32
  tl.param softmax_scale : tl.float32
  q_seq_len = $(p.length)
  start_m = tl.program_id(0)
  off_bh = tl.program_id(1)
  channel_block = tl.program_id(2)
  off_h = off_bh - (off_bh // $(p.heads)) * $(p.heads)
  off_b = off_bh // $(p.heads)
  head_groups = $(p.heads) // $(p.kvHeads)
  off_h_kv = off_h // head_groups
  Q += off_b * $(p.qb) + off_h * $(p.qh)
  K += off_b * $(p.kb) + off_h_kv * $(p.kh)
  V += off_b * $(p.vb) + off_h_kv * $(p.vh)
  offs_m = start_m * $(16) + tl.arange(0, 16)
  offs_n = tl.arange(0, 16)
  offs_d = tl.arange(0, 16)
  off_q = offs_m[:, None] * $(p.qm) + offs_d[None, :] * $(p.qd)
  off_k = offs_n[None, :] * $(p.kn) + offs_d[:, None] * $(p.kd)
  off_v = offs_n[:, None] * $(p.vn) + (channel_block * $(16) + offs_d[None, :]) * $(p.vd)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_i = tl.zeros([16], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([16], dtype=tl.float32)
  acc = tl.zeros([16,16], dtype=tl.float32)
  layout_h = off_h % $(p.layouts)
  layout_ptr = layout_csr_row_indices + layout_h * $(p.rowStride) + start_m
  start_l = (tl.load(layout_ptr)).to(tl.int32)
  end_l = (tl.load(layout_ptr + $(1))).to(tl.int32)
  for col_idx_idx in range(start_l, end_l) {
    col_idx = (tl.load(layout_csr_col_indices +
      layout_h * $(p.colStride) + col_idx_idx)).to(tl.int32)
    start_n = col_idx * $(16)
    qk = tl.zeros([16,16], dtype=tl.float32)
    for score_block in tl.static_range($(p.blocks)) {
      if EVEN_M {
        q = tl.load(q_ptrs + score_block * $(16) * $(p.qd))
      } else {
        q = tl.load(q_ptrs + score_block * $(16) * $(p.qd),
          mask=offs_m[:, None] < q_seq_len, other=0)
      }
      if EVEN_N {
        k = tl.load(k_ptrs + start_n * $(p.kn) + score_block * $(16) * $(p.kd))
      } else {
        k = tl.load(k_ptrs + start_n * $(p.kn) + score_block * $(16) * $(p.kd),
          mask=offs_n[None, :] + start_n < $(p.length), other=0)
      }
      qk += tl.dot(q, k)
    }
    qk *= softmax_scale
    qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
    m_ij = tl.max(qk, 1)
    p = tl.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    m_i_new = tl.maximum(m_i, m_ij)
    alpha = tl.exp(m_i - m_i_new)
    beta = tl.exp(m_ij - m_i_new)
    l_i_new = alpha * l_i + beta * l_ij
    p_scale = beta / l_i_new
    p = p * p_scale[:, None]
    acc_scale = l_i / l_i_new * alpha
    acc = acc * acc_scale[:, None]
    p = (p).to(tl.float32)
    if EVEN_N {
      v = tl.load(v_ptrs + start_n * $(p.vn))
    } else {
      v = tl.load(v_ptrs + start_n * $(p.vn),
        mask=offs_n[:, None] + start_n < $(p.length), other=0)
    }
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = off_b * $(p.ob) + off_h * $(p.oh) +
    offs_m[:, None] * $(p.om) + channel_block * $(16) + offs_d[None, :]
  out_ptrs = out + off_o
  tl.store(out_ptrs, acc, mask=offs_m[:, None] < q_seq_len)
}

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.BlockSparseAttn.KernelOnly

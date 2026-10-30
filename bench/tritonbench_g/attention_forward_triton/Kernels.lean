import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit

/-! Complete repaired FP32 caller source; half P/V and half PV-result remain explicit.
The Python helper is inlined and unused stage/shape arguments are retained. -/
namespace VeriTile.Bench.Repairs.AttentionForwardTriton.TypedSource
open VeriTile.Triton

def repaired
    (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel :=
  let Q : Region .fp32 := q.cast
  let K : Region .fp32 := k.cast
  let V : Region .fp32 := v.cast
  let Q_scale : Region .fp32 := qScale.cast
  let K_scale : Region .fp32 := kScale.cast
  let Out : Region .fp16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_h).to(tl.int64) * $(stride_kh)
  v_offset = (off_z).to(tl.int64) * $(stride_vz) + (off_h).to(tl.int64) * $(stride_vh)
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + k_offset + offs_k[:, None] * $(stride_kk) + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + v_offset + offs_n[:, None] * $(stride_vk) + offs_k[None, :] * $(stride_vn)
  O_block_ptr = Out + o_offset + offs_m[:, None] * $(stride_om) + offs_k[None, :] * $(stride_on)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :], other=0.0)
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[:, None]
    k = tl.load(K_ptrs, mask=k_mask, other=0.0)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    qk = tl.where(offs_n[None, :] < ($(N_CTX) - start_n), qk, -float("inf"))
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :], other=0.0)
    v = (v).to(tl.float16)
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(stride_kn)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  acc = acc / l_i[:, None]
  -- Out.type.element_ty is fp16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.float16),
    mask=(offs_m[:, None] < $(N_CTX)) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_ACTIVE))[None, :])
}


/-- Hoist the loop-invariant active-head mask and reuse it in every Q/K/V
load and the final store. Floating arithmetic and memory footprints are unchanged. -/
def hoistedHeadMask
    (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL HEAD_ACTIVE _STAGE : Nat) :
    ComputeKernel :=
  let Q : Region .fp32 := q.cast
  let K : Region .fp32 := k.cast
  let V : Region .fp32 := v.cast
  let Q_scale : Region .fp32 := qScale.cast
  let K_scale : Region .fp32 := kScale.cast
  let Out : Region .fp16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_h).to(tl.int64) * $(stride_kh)
  v_offset = (off_z).to(tl.int64) * $(stride_vz) + (off_h).to(tl.int64) * $(stride_vh)
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  q_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_M))
  k_scale_offset = off_hz * tl.cdiv($(N_CTX), $(BLOCK_N))

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_k = tl.arange(0, $(BLOCK_DMODEL))
  head_mask = offs_k < $(HEAD_ACTIVE)
  Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
  Q_scale_ptr = Q_scale + q_scale_offset + start_m
  K_ptrs = K + k_offset + offs_k[:, None] * $(stride_kk) + offs_n[None, :] * $(stride_kn)
  K_scale_ptr = K_scale + k_scale_offset
  V_ptrs = V + v_offset + offs_n[:, None] * $(stride_vk) + offs_k[None, :] * $(stride_vn)
  O_block_ptr = Out + o_offset + offs_m[:, None] * $(stride_om) + offs_k[None, :] * $(stride_on)
  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(Q_ptrs,
    mask=(offs_m[:, None] < $(N_CTX)) & head_mask[None, :], other=0.0)
  q_scale = tl.load(Q_scale_ptr)
  for start_n in range(0, $(N_CTX), $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    k_mask = (offs_n[None, :] < ($(N_CTX) - start_n)) &
      head_mask[:, None]
    k = tl.load(K_ptrs, mask=k_mask, other=0.0)
    k_scale = tl.load(K_scale_ptr)
    qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
    qk = tl.where(offs_n[None, :] < ($(N_CTX) - start_n), qk, -float("inf"))
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk = qk - m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    v = tl.load(V_ptrs,
      mask=(offs_n[:, None] < ($(N_CTX) - start_n)) &
        head_mask[None, :], other=0.0)
    v = (v).to(tl.float16)
    p = (p).to(tl.float16)
    acc += tl.dot(p, v, out_dtype=tl.float16)
    m_i = m_ij
    K_ptrs += $(BLOCK_N) * $(stride_kn)
    K_scale_ptr += $(1)
    V_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  acc = acc / l_i[:, None]
  -- Out.type.element_ty is fp16 in the Python wrapper.
  tl.store(O_block_ptr, (acc).to(tl.float16),
    mask=(offs_m[:, None] < $(N_CTX)) & head_mask[None, :])
}



#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionForwardTriton.TypedSource

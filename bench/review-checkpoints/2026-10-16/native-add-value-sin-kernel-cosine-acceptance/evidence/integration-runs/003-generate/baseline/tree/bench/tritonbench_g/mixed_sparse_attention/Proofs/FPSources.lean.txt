import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false
/-- Complete source under its previously accepted fp16 storage configuration.
The scalar scale argument is explicitly fp32. Scalar cond[:,None] broadcasts
are represented at their operand shape, preserving the same lane predicate.
All eight fixed block iterations remain, including masked metadata undef. -/
def source
    (Q K V : RegionName) (seqlens : Region .nat) (sm_scale : ℝ)
    (block_count block_offset column_count column_index : Region .nat)
    (Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vn stride_vk
      stride_oz stride_oh stride_om stride_ok
      Z H N_CTX NUM_ROWS NNZ_S NNZ_V
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat)
    :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  seqlen = tl.load(seqlens + off_hz // $(H))
  if start_m * $(BLOCK_M) >= seqlen {
    return
  }

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))

  qo_offset = (off_hz // $(H)) * $(stride_qz) + (off_hz % $(H)) * $(stride_qh)
  kv_offset = (off_hz // $(H)) * $(stride_kz) + (off_hz % $(H)) * $(stride_kh)

  q_ptrs = Q + qo_offset + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  k_ptrs = K + kv_offset + offs_d[:, None] * $(stride_kk)
  v_ptrs = V + kv_offset + offs_d[None, :] * $(stride_vk)
  o_ptrs = Out + qo_offset + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_ok)

  num_blks = tl.load(block_count + off_hz * $(NUM_ROWS) + start_m)
  blks_ptr = block_offset + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_S)
  num_cols = tl.load(column_count + off_hz * $(NUM_ROWS) + start_m)
  cols_ptr = column_index + (off_hz * $(NUM_ROWS) + start_m) * $(NNZ_V)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = tl.full([], $((sm_scale : ℝ)), dtype=tl.float32) * 1.44269504
  q = tl.load(q_ptrs)
  q = (q * qk_scale).to(tl.float16)

  m_mask = offs_m[:, None] < seqlen

  max_num_blks = $(8)
  for block_index in range(max_num_blks) {
    cond = block_index < num_blks
    start_n = tl.load(blks_ptr + block_index, mask=cond)
    cols = start_n + offs_n
    n_mask = (cols < seqlen) & cond
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    causal_mask = cols[None, :] <= offs_m[:, None]
    qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(tl.float16), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  max_num_cols = $(16)
  for start_n in range($(0), max_num_cols, $(BLOCK_N)) {
    cond = start_n < num_cols
    n_mask = (start_n + offs_n < num_cols) & cond
    cols = tl.load(cols_ptr + start_n + offs_n, mask=cond[:, None], other=0)
    k = tl.load(k_ptrs + cols[None, :] * $(stride_kn), mask=n_mask[None, :], other=0.0)
    v = tl.load(v_ptrs + cols[:, None] * $(stride_vn), mask=n_mask[:, None], other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk = tl.where(m_mask & n_mask, qk, float("-inf"))
    qk += tl.dot(q, k)
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc_scale = l_i * 0 + alpha
    acc *= acc_scale[:, None]
    acc += tl.dot((p).to(tl.float16), v)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
  }

  acc /= l_i[:, None]
  tl.store(o_ptrs, (acc).to(tl.float16), mask=m_mask)
}

end
end VeriTile.Bench.TritonBenchG.MixedSparseAttention.FPStages

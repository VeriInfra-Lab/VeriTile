import VeriTile.Triton.DSL
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

/-! Full readable FP32 selected source and directly related candidates. The
historical original and scale-first transcriptions are preserved by complete
source-function identities in Proofs/ReadableSource. Integer metadata, finite
causal sentinel, guarded normalization and all masks remain explicit. -/
namespace VeriTile.Bench.Repairs.ContextAttnBloomSource
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000

def kernel :=
  fun 
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  => triton {
  tl.ports inputs=[Q,K,V,B_Start_Loc,B_Seqlen,Req_to_tokens,B_req_idx,b_prompt_cache_len], outputs=[Out]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32, Req_to_tokens = tl.int32,
    B_req_idx = tl.int32, b_prompt_cache_len = tl.int32

  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($((0 : Int)), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

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
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(tl.float32)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  -- Triton implicitly converts the fp32 accumulator to Out element storage.
  tl.store(out_ptrs, (acc).to(tl.float32),
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))

}

/-- Existing multiplication operand orientation candidate. -/
def scaleFirst :=
  fun 
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  => triton {
  tl.ports inputs=[Q,K,V,B_Start_Loc,B_Seqlen,Req_to_tokens,B_req_idx,b_prompt_cache_len], outputs=[Out]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32, Req_to_tokens = tl.int32,
    B_req_idx = tl.int32, b_prompt_cache_len = tl.int32

  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($((0 : Int)), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = $((sm_scale : ℝ)) * qk
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

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
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(tl.float32)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  -- Triton implicitly converts the fp32 accumulator to Out element storage.
  tl.store(out_ptrs, (acc).to(tl.float32),
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))

}

/-- Reuse the already computed block row origin, eliminating a repeated
integer multiplication. Every FP operation is unchanged. -/
def indexShared :=
  fun 
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  => triton {
  tl.ports inputs=[Q,K,V,B_Start_Loc,B_Seqlen,Req_to_tokens,B_req_idx,b_prompt_cache_len], outputs=[Out]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32, Req_to_tokens = tl.int32,
    B_req_idx = tl.int32, b_prompt_cache_len = tl.int32

  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_kv_head = cur_head // $(kv_group_num)

  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

  block_start_loc = $(BLOCK_M) * start_m

  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  offs_m = block_start_loc + tl.arange(0, $(BLOCK_M))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_qbs) +
    cur_head * $(stride_qh) + offs_d[None, :] * $(stride_qd)

  q = tl.load(Q + off_q,
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)),
    other=0.0)

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum((start_m + $(1)) * $(BLOCK_M) + prompt_cache_len,
    cur_batch_seq_len + prompt_cache_len)

  for start_n in range($((0 : Int)), block_mask * block_end_loc, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    kv_loc = tl.load(Req_to_tokens + $(stride_req_to_tokens_b) * cur_batch_req_idx +
      $(stride_req_to_tokens_s) * (start_n + offs_n),
      mask=(start_n + offs_n) < block_end_loc,
      other=0)
    off_k = kv_loc[None, :] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[:, None] * $(stride_kd)
    k = tl.load(K + off_k,
      mask=((start_n + offs_n[None, :]) < block_end_loc) &
        (offs_d[:, None] < $(head_dim)),
      other=0.0)

    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((sm_scale : ℝ))
    qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :],
      qk, -100000000.0)

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
    acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
    acc = acc * acc_scale[:, None]
    off_v = kv_loc[:, None] * $(stride_vbs) + cur_kv_head * $(stride_vh) +
      offs_d[None, :] * $(stride_vd)
    v = tl.load(V + off_v,
      mask=((start_n + offs_n[:, None]) < block_end_loc) &
        (offs_d[None, :] < $(head_dim)),
      other=0.0)
    p = (p).to(tl.float32)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(stride_obs) +
    cur_head * $(stride_oh) + offs_d[None, :] * $(stride_od)
  out_ptrs = Out + off_o
  -- Triton implicitly converts the fp32 accumulator to Out element storage.
  tl.store(out_ptrs, (acc).to(tl.float32),
    mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < $(head_dim)))

}

end
end VeriTile.Bench.Repairs.ContextAttnBloomSource

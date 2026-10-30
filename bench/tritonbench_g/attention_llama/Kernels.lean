import bench.tritonbench_g.attention_llama.Sources.Kernels
import VeriTile.Triton.Source
import VeriTile.Triton.Float.Structural

/-! A nonexecuting FP32 compute-carrier parameter declaration retains the
actual raw scalar ABI binding (including integer and constexpr-one paths).
All actual kernel statements remain in their ordinary quotations.

Separate repair: global-key masking after scaling. Both source dtype paths,
ordered two-argument dots plus additions, FP32 state, half probability cast,
causal branch and output store are retained. -/
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TypedSource
open VeriTile.Triton

def half
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  tl.param sm_scale : tl.float32
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= sm_scale
    qk = tl.where(block_n_offs[None, :] < $(N_CTX), qk, float("-inf"))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}

def bytes
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.int8, V = tl.int8, Out = tl.float16
  tl.param sm_scale : tl.float32
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    k = k.to(tl.float8e5, bitcast=True)
    k = k.to(tl.float16)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= sm_scale
    qk = tl.where(block_n_offs[None, :] < $(N_CTX), qk, float("-inf"))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = tl.sum(p, 1) + l_prev
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    v = v.to(tl.float8e5, bitcast=True)
    v = v.to(tl.float16)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}
/-- Existing carry-first normalizer candidate, retaining the runtime scalar
ABI, both genuine storage paths, source precision, masks and ordered dots. -/
def carryFirstHalf
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, Out = tl.float16
  tl.param sm_scale : tl.float32
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= sm_scale
    qk = tl.where(block_n_offs[None, :] < $(N_CTX), qk, float("-inf"))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = l_prev + tl.sum(p, 1)
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}

def carryFirstBytes
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.int8, V = tl.int8, Out = tl.float16
  tl.param sm_scale : tl.float32
  start_m = tl.program_id(0)

  head_idx = tl.program_id(1)
  batch_id = head_idx // $(N_HEAD)
  off_hz = head_idx % $(N_HEAD)

  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_q = batch_id * $(stride_qz) + off_hz * $(stride_qh) + offs_m[:, None] * $(stride_qm) + offs_d[None, :] * $(stride_qk)
  off_k = batch_id * $(stride_kz) + off_hz * $(stride_kh) + offs_n[:, None] * $(stride_kn) + offs_d[None, :] * $(stride_kk)
  off_v = batch_id * $(stride_vz) + off_hz * $(stride_vh) + offs_n[:, None] * $(stride_vk) + offs_d[None, :] * $(stride_vn)
  q_ptrs = Q + off_q
  k_ptrs = K + off_k
  v_ptrs = V + off_v
  m_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_prev = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  q = tl.load(q_ptrs, offs_m[:, None] < $(H), other=0.0)
  block_n_end = $(N_CTX)
  if IS_CAUSAL {
    block_n_end = (start_m + $(1)) * $(BLOCK_N) + $(start_position)
  }
  for start_n in range($(0), block_n_end, $(BLOCK_N)) {
    block_n_offs = start_n + offs_n
    k = tl.load(k_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    k = k.to(tl.float8e5, bitcast=True)
    k = k.to(tl.float16)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, tl.trans(k))
    qk *= sm_scale
    qk = tl.where(block_n_offs[None, :] < $(N_CTX), qk, float("-inf"))
    if IS_CAUSAL {
      qk = tl.where(offs_m[:, None] >= (block_n_offs[None, :] + $(start_position)), qk, float("-inf"))
    }
    m_curr = tl.maximum(tl.max(qk, 1), m_prev)
    l_prev *= tl.exp(m_prev - m_curr)
    p = tl.exp(qk - m_curr[:, None])
    l_curr = l_prev + tl.sum(p, 1)
    l_rcp = 1.0 / l_curr
    p *= l_rcp[:, None]
    acc *= (l_prev * l_rcp)[:, None]
    p = (p).to(tl.float16)
    v = tl.load(v_ptrs, block_n_offs[:, None] < $(N_CTX), other=0.0)
    v = v.to(tl.float8e5, bitcast=True)
    v = v.to(tl.float16)
    acc += tl.dot(p, v)
    l_prev = l_curr
    m_prev = m_curr
    k_ptrs += $(BLOCK_N) * $(stride_kn)
    v_ptrs += $(BLOCK_N) * $(stride_vk)
  }
  start_m = tl.program_id(0)
  offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))

  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_o = batch_id * $(stride_oz) + off_hz * $(stride_oh) + offs_m[:, None] * $(stride_om) + offs_d[None, :] * $(stride_on)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float16), offs_m[:, None] < $(H))
}
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TypedSource

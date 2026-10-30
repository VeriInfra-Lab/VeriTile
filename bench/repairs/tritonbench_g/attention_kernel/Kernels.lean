import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource
open VeriTile.Triton
inductive Storage where | fp16 | bf16 deriving DecidableEq
def Storage.float : Storage → FloatDType | .fp16=>.fp16 | .bf16=>.bf16
def Storage.tile (s : Storage) : TileDType := s.float.toTileDType
namespace FP16

def repaired
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      stride_b0z stride_b0h stride_b0m
      _Z H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .fp16 := q.cast
  let K : Region .fp16 := k.cast
  let V : Region .fp16 := v.cast
  let B0 : Region .fp16 := bias.cast
  let Out : Region .fp16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  q_offset = off_z * $(stride_qz) + off_h * $(stride_qh)
  k_offset = off_z * $(stride_kz) + off_h * $(stride_kh)
  v_offset = off_z * $(stride_vz) + off_h * $(stride_vh)
  o_offset = off_z * $(stride_oz) + off_h * $(stride_oh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = (tl.arange(0, $(BLOCK_M))).to(tl.int64)
  b_offset = off_z * $(stride_b0z) + off_h * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :],
    mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr, boundary_check=([1] : List Nat), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float16)
    -- The default dot result is FP32; expose its existing compute carrier.
    qk += (tl.dot(q, k)).to(tl.float32)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N),
      mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.float16))
    qk = tl.where((tl.arange(0, $(BLOCK_N)))[None, :] < ($(N_CTX) - start_n), qk, -float("inf"))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += (tl.dot((p).to(tl.float16), v)).to(tl.float32)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.float16), boundary_check=([0] : List Nat))
}

/-- Issue the unchanged Q load/scale before independent pointer and carry setup. -/
def queryEarly
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      stride_b0z stride_b0h stride_b0m
      _Z H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .fp16 := q.cast
  let K : Region .fp16 := k.cast
  let V : Region .fp16 := v.cast
  let B0 : Region .fp16 := bias.cast
  let Out : Region .fp16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  q_offset = off_z * $(stride_qz) + off_h * $(stride_qh)
  k_offset = off_z * $(stride_kz) + off_h * $(stride_kh)
  v_offset = off_z * $(stride_vz) + off_h * $(stride_vh)
  o_offset = off_z * $(stride_oz) + off_h * $(stride_oh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
  q = (q * qk_scale).to(tl.float16)
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = (tl.arange(0, $(BLOCK_M))).to(tl.int64)
  b_offset = off_z * $(stride_b0z) + off_h * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :],
    mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr, boundary_check=([1] : List Nat), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float16)
    -- The default dot result is FP32; expose its existing compute carrier.
    qk += (tl.dot(q, k)).to(tl.float32)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N),
      mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.float16))
    qk = tl.where((tl.arange(0, $(BLOCK_N)))[None, :] < ($(N_CTX) - start_n), qk, -float("inf"))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += (tl.dot((p).to(tl.float16), v)).to(tl.float32)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.float16), boundary_check=([0] : List Nat))
}

end FP16

namespace BF16

def repaired
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      stride_b0z stride_b0h stride_b0m
      _Z H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .bf16 := q.cast
  let K : Region .bf16 := k.cast
  let V : Region .bf16 := v.cast
  let B0 : Region .bf16 := bias.cast
  let Out : Region .bf16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  q_offset = off_z * $(stride_qz) + off_h * $(stride_qh)
  k_offset = off_z * $(stride_kz) + off_h * $(stride_kh)
  v_offset = off_z * $(stride_vz) + off_h * $(stride_vh)
  o_offset = off_z * $(stride_oz) + off_h * $(stride_oh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
  q = (q * qk_scale).to(tl.bfloat16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = (tl.arange(0, $(BLOCK_M))).to(tl.int64)
  b_offset = off_z * $(stride_b0z) + off_h * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :],
    mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr, boundary_check=([1] : List Nat), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.bfloat16)
    -- The default dot result is FP32; expose its existing compute carrier.
    qk += (tl.dot(q, k)).to(tl.float32)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N),
      mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.bfloat16))
    qk = tl.where((tl.arange(0, $(BLOCK_N)))[None, :] < ($(N_CTX) - start_n), qk, -float("inf"))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += (tl.dot((p).to(tl.bfloat16), v)).to(tl.float32)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.bfloat16), boundary_check=([0] : List Nat))
}

/-- Issue the unchanged Q load/scale before independent pointer and carry setup. -/
def queryEarly
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      stride_b0z stride_b0h stride_b0m
      _Z H N_CTX P_SEQ BIAS_LAST_SIZE _B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    : ComputeKernel :=
  let Q : Region .bf16 := q.cast
  let K : Region .bf16 := k.cast
  let V : Region .bf16 := v.cast
  let B0 : Region .bf16 := bias.cast
  let Out : Region .bf16 := out.cast
  triton {
  start_m = tl.program_id(0)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  q_offset = off_z * $(stride_qz) + off_h * $(stride_qh)
  k_offset = off_z * $(stride_kz) + off_h * $(stride_kh)
  v_offset = off_z * $(stride_vz) + off_h * $(stride_vh)
  o_offset = off_z * $(stride_oz) + off_h * $(stride_oh)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  qk_scale = $((sm_scale : ℝ)) * 1.44269504
  q = tl.load(Q_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
  q = (q * qk_scale).to(tl.bfloat16)
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($((BLOCK_DMODEL : Nat)), $((N_CTX + P_SEQ : Nat))),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  V_block_ptr = tl.make_block_ptr(base=V + v_offset,
    shape=($((N_CTX + P_SEQ : Nat)), $(BLOCK_DMODEL)),
    strides=($(stride_vk), $(stride_vn)),
    offsets=(0, 0),
    block_shape=($(BLOCK_N), $(BLOCK_DMODEL)),
    order=(1, 0))

  m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = (tl.arange(0, $(BLOCK_M))).to(tl.int64)
  b_offset = off_z * $(stride_b0z) + off_h * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :],
    mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr, boundary_check=([1] : List Nat), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=([0] : List Nat), padding_option="zero")
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.bfloat16)
    -- The default dot result is FP32; expose its existing compute carrier.
    qk += (tl.dot(q, k)).to(tl.float32)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N),
      mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0.0)
    -- A floating scalar does not widen the bias tensor in Triton.
    -- Resolve its implicit conversion to the storage dtype explicitly.
    qk += ((b0 + b1) * (1.44269504).to(tl.bfloat16))
    qk = tl.where((tl.arange(0, $(BLOCK_N)))[None, :] < ($(N_CTX) - start_n), qk, -float("inf"))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += (tl.dot((p).to(tl.bfloat16), v)).to(tl.float32)
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new
    K_block_ptr = tl.advance(K_block_ptr, [$(0), $(BLOCK_N)])
    V_block_ptr = tl.advance(V_block_ptr, [$(BLOCK_N), $(0)])
  }

  acc = acc / l_i[:, None]
  O_block_ptr = tl.make_block_ptr(base=Out + o_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_om), $(stride_on)),
    offsets=(start_m * $(BLOCK_M), 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  tl.store(O_block_ptr, (acc).to(tl.bfloat16), boundary_check=([0] : List Nat))
}

end BF16

def repaired (storage : Storage) :=
  match storage with | .fp16=>FP16.repaired | .bf16=>BF16.repaired
def queryEarly (storage : Storage) :=
  match storage with | .fp16=>FP16.queryEarly | .bf16=>BF16.queryEarly
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.AttentionKernel.Whole001.TypedSource

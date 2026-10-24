import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Full
open VeriTile.Triton FP.Structural
noncomputable section

/-- Repaired independent logical view metadata. The public wrapper selects
standard power-of-two widths and P_SEQ=0; both precision branches share this metadata.
The final four fields are integer-only launch and decoded compiler-layout metadata. -/
structure Parameters where
  q : RegionName
  k : RegionName
  v : RegionName
  bias : RegionName
  output : RegionName
  scale : ℝ
  /-- Exact observed FP32 launch word, including signed zero. -/
  scaleWord : BitVec 32
  qBatchStride : Nat
  qHeadStride : Nat
  qRowStride : Nat
  qColStride : Nat
  kBatchStride : Nat
  kHeadStride : Nat
  kSeqStride : Nat
  kDimStride : Nat
  vBatchStride : Nat
  vHeadStride : Nat
  vSeqStride : Nat
  vDimStride : Nat
  outBatchStride : Nat
  outHeadStride : Nat
  outRowStride : Nat
  outColStride : Nat
  biasBatchStride : Nat
  biasHeadStride : Nat
  biasRowStride : Nat
  batches : Nat
  heads : Nat
  length : Nat
  blockM : Nat
  blockN : Nat
  head : Nat
  biasLast : Nat
  biasNumel : Nat
  numWarps : Nat := 4
  numStages : Nat := 2
  nativeColumns : Nat := 1
  nativeSerial : Bool := Bool.false

/-- `range(0, N_CTX, BLOCK_N)` visits the final partial block in full. -/
def Parameters.blocks (p : Parameters) : Nat :=
  (p.length + p.blockN - 1) / p.blockN

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Full
namespace VeriTile.Bench.Repairs.AttentionKernelAligned.Source
open VeriTile.Triton FP.Structural
open Full
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
def typedSurface
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      stride_b0z stride_b0h stride_b0m
      _Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    :
    ComputeKernel := triton {
  tl.region Q = tl.float16, K = tl.float16, V = tl.float16, B0 = tl.float16, Out = tl.float16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_b = off_hz // $(H)
  off_h = off_hz % $(H)
  q_offset = off_b * $(stride_qz) + off_h * $(stride_qh)
  k_offset = off_b * $(stride_kz) + off_h * $(stride_kh)
  v_offset = off_b * $(stride_vz) + off_h * $(stride_vh)
  o_offset = off_b * $(stride_oz) + off_h * $(stride_oh)
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
  q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  q = (q * qk_scale).to(tl.float16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_b * $(stride_b0z) + off_h * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :],
    mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0)
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float16)
    qk += tl.dot(q, k, out_dtype=tl.float16)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N),
      mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0)
    qk += (b0 + b1)
    qk = tl.where((start_n + tl.arange(0, $(BLOCK_N)))[None, :] < $(N_CTX), qk, (float("-inf")).to(tl.float16))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(tl.float16), v)
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
  tl.store(O_block_ptr, (acc).to(tl.float16), boundary_check=(0, 1))
}


def source (p : Parameters) : ComputeKernel :=
  typedSurface p.q p.k p.v p.bias p.output p.scale
    p.qBatchStride p.qHeadStride p.qRowStride p.qColStride
    p.kBatchStride p.kHeadStride p.kSeqStride p.kDimStride
    p.vBatchStride p.vHeadStride p.vSeqStride p.vDimStride
    p.outBatchStride p.outHeadStride p.outRowStride p.outColStride
    p.biasBatchStride p.biasHeadStride p.biasRowStride p.batches p.heads
    p.length 0 p.biasLast p.biasNumel p.head p.blockM p.blockN

def prologue (p : Parameters) := (source p).surfaceBody.take 23
def body (p : Parameters) : List ComputeStmt :=
  match ((source p).surfaceBody.drop 23).head? with
  | some (.forRangeDyn _ _ _ _ b) => b
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 24

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.Source

namespace VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionKernelAligned.Full
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
def typedSurface
    (Q K V B0 Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      stride_b0z stride_b0h stride_b0m
      _Z H N_CTX P_SEQ BIAS_LAST_SIZE B0_NUMEL
      BLOCK_DMODEL BLOCK_M BLOCK_N : Nat)
    :
    ComputeKernel := triton {
  tl.region Q = tl.bfloat16, K = tl.bfloat16, V = tl.bfloat16, B0 = tl.bfloat16, Out = tl.bfloat16
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_b = off_hz // $(H)
  off_h = off_hz % $(H)
  q_offset = off_b * $(stride_qz) + off_h * $(stride_qh)
  k_offset = off_b * $(stride_kz) + off_h * $(stride_kh)
  v_offset = off_b * $(stride_vz) + off_h * $(stride_vh)
  o_offset = off_b * $(stride_oz) + off_h * $(stride_oh)
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
  q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
  q = (q * qk_scale).to(tl.bfloat16)
  lo = 0
  hi = $((N_CTX + P_SEQ : Nat))

  b_ptr_offsets_m = tl.arange(0, $(BLOCK_M))
  b_offset = off_b * $(stride_b0z) + off_h * $(stride_b0h)
  b_ptr_offsets_n_1 = (tl.arange(0, $(BLOCK_N)) % $(BIAS_LAST_SIZE)) +
    $(BIAS_LAST_SIZE)
  b1 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
    $(stride_b0m))[:, None] + b_ptr_offsets_n_1[None, :],
    mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0)
  for start_n in range(lo, hi, $(BLOCK_N)) {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.bfloat16)
    qk += (tl.dot(q, k, out_dtype=tl.float32)).to(tl.bfloat16)

    b0 = tl.load(B0 + b_offset + ((start_m * $(BLOCK_M) + b_ptr_offsets_m) *
      $(stride_b0m))[:, None] + start_n // $(BLOCK_N),
      mask=(start_m * $(BLOCK_M) + b_ptr_offsets_m)[:, None] < $(N_CTX), other=0)
    qk += (b0 + b1)
    qk = tl.where((start_n + tl.arange(0, $(BLOCK_N)))[None, :] < $(N_CTX), qk, (float("-inf")).to(tl.bfloat16))

    m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    alpha = tl.math.exp2(m_i - m_i_new)
    p = tl.math.exp2(qk - m_i_new[:, None])
    acc *= alpha[:, None]
    acc += tl.dot((p).to(tl.bfloat16), v)
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
  tl.store(O_block_ptr, (acc).to(tl.bfloat16), boundary_check=(0, 1))
}


def source (p : Parameters) : ComputeKernel :=
  typedSurface p.q p.k p.v p.bias p.output p.scale
    p.qBatchStride p.qHeadStride p.qRowStride p.qColStride
    p.kBatchStride p.kHeadStride p.kSeqStride p.kDimStride
    p.vBatchStride p.vHeadStride p.vSeqStride p.vDimStride
    p.outBatchStride p.outHeadStride p.outRowStride p.outColStride
    p.biasBatchStride p.biasHeadStride p.biasRowStride p.batches p.heads
    p.length 0 p.biasLast p.biasNumel p.head p.blockM p.blockN

def prologue (p : Parameters) := (source p).surfaceBody.take 23
def body (p : Parameters) : List ComputeStmt :=
  match ((source p).surfaceBody.drop 23).head? with
  | some (.forRangeDyn _ _ _ _ b) => b
  | _ => []
def epilogue (p : Parameters) := (source p).surfaceBody.drop 24

end
end VeriTile.Bench.Repairs.AttentionKernelAligned.BF16.Source

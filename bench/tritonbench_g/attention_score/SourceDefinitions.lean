import VeriTile.Triton.DSL

/-! The repaired attention-score source and its scalar-left multiplication
comparison. Every proof consumes these same bodies, including integer casts. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.TypedSource
open VeriTile.Triton

def original
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel := triton {
  tl.ports inputs=[Q, K, M], outputs=[Out]
  tl.region Q = tl.float32, K = tl.float32, M = tl.float32, Out = tl.float32
  start_n = (tl.program_id(0)).to(tl.int64)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + (tl.arange(0, $(BLOCK_M))).to(tl.int64)
  o = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(0, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, (start_n * $(BLOCK_N)).to(tl.int32)),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  if IS_EVEN_N {
    k = tl.load(K_block_ptr)
  } else {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  lo = 0
  hi = ($(N_CTX)).to(tl.int64)
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  for start_m in range(lo, hi, $(BLOCK_M)) {
    start_m = tl.multiple_of(start_m, $(BLOCK_M))
    if IS_EVEN_M {
      q = tl.load(Q_block_ptr)
    } else {
      q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    m = tl.load(m_ptrs, mask=start_m + tl.arange(0, $(BLOCK_M)) < $(N_CTX), other=0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk * qk_scale
    if SLIDING_WINDOW {
      dist = (tl.arange(0, $(BLOCK_M))[:, None]).to(tl.int64) -
        (tl.arange(0, $(BLOCK_N))[None, :]).to(tl.int64) + start_m -
        start_n * $(BLOCK_N) + ($((sliding_window_offset : Int))).to(tl.int64)
      if COMPLEMENT_SLIDING_WINDOW {
        mask = dist >= ($((sliding_window_size : Int))).to(tl.int64)
      } else {
        mask = (dist >= $(0)) & (dist < ($((sliding_window_size : Int))).to(tl.int64))
      }
    }
    qk = qk - m[:, None]
    p = tl.math.exp2(qk)
    if SLIDING_WINDOW {
      p = tl.where(mask, p, 0)
    }
    if not IS_EVEN_M {
      p = tl.where(((tl.arange(0, $(BLOCK_M)) + start_m) < $(N_CTX))[:, None],
        p, 0)
    }
    o += tl.sum(p, axis=0)
    Q_block_ptr = tl.advance(Q_block_ptr, offsets=($(BLOCK_M), 0))
    m_ptrs = m_ptrs + $(BLOCK_M)
  }
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  o_range = (tl.arange(0, $(BLOCK_N))).to(tl.int64) + start_n * $(BLOCK_N)
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, (o).to(tl.float32),
    mask=o_range < $(NKV_CTX))
}


/-- Only score multiplication operand order changes. -/
def scalarLeft
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh _stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel := triton {
  tl.ports inputs=[Q, K, M], outputs=[Out]
  tl.region Q = tl.float32, K = tl.float32, M = tl.float32, Out = tl.float32
  start_n = (tl.program_id(0)).to(tl.int64)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // $(H)
  off_h = off_hz % $(H)
  off_hkv = off_h // ($(H) // $(H_KV))
  q_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
  k_offset = (off_z).to(tl.int64) * $(stride_kz) + (off_hkv).to(tl.int64) * $(stride_kh)
  m_ptrs = M + off_hz * $(ROUND_CTX) + (tl.arange(0, $(BLOCK_M))).to(tl.int64)
  o = tl.zeros([$(BLOCK_M)], dtype=tl.float32)
  Q_block_ptr = tl.make_block_ptr(base=Q + q_offset,
    shape=($(N_CTX), $(BLOCK_DMODEL)),
    strides=($(stride_qm), $(stride_qk)),
    offsets=(0, 0),
    block_shape=($(BLOCK_M), $(BLOCK_DMODEL)),
    order=(1, 0))
  K_block_ptr = tl.make_block_ptr(base=K + k_offset,
    shape=($(BLOCK_DMODEL), $(NKV_CTX)),
    strides=($(stride_kk), $(stride_kn)),
    offsets=(0, (start_n * $(BLOCK_N)).to(tl.int32)),
    block_shape=($(BLOCK_DMODEL), $(BLOCK_N)),
    order=(0, 1))
  if IS_EVEN_N {
    k = tl.load(K_block_ptr)
  } else {
    k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
  }
  lo = 0
  hi = ($(N_CTX)).to(tl.int64)
  qk_scale = $((sm_scale : ℝ))
  qk_scale *= 1.4426950408889634
  for start_m in range(lo, hi, $(BLOCK_M)) {
    start_m = tl.multiple_of(start_m, $(BLOCK_M))
    if IS_EVEN_M {
      q = tl.load(Q_block_ptr)
    } else {
      q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    }
    m = tl.load(m_ptrs, mask=start_m + tl.arange(0, $(BLOCK_M)) < $(N_CTX), other=0)
    qk = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk = qk_scale * qk
    if SLIDING_WINDOW {
      dist = (tl.arange(0, $(BLOCK_M))[:, None]).to(tl.int64) -
        (tl.arange(0, $(BLOCK_N))[None, :]).to(tl.int64) + start_m -
        start_n * $(BLOCK_N) + ($((sliding_window_offset : Int))).to(tl.int64)
      if COMPLEMENT_SLIDING_WINDOW {
        mask = dist >= ($((sliding_window_size : Int))).to(tl.int64)
      } else {
        mask = (dist >= $(0)) & (dist < ($((sliding_window_size : Int))).to(tl.int64))
      }
    }
    qk = qk - m[:, None]
    p = tl.math.exp2(qk)
    if SLIDING_WINDOW {
      p = tl.where(mask, p, 0)
    }
    if not IS_EVEN_M {
      p = tl.where(((tl.arange(0, $(BLOCK_M)) + start_m) < $(N_CTX))[:, None],
        p, 0)
    }
    o += tl.sum(p, axis=0)
    Q_block_ptr = tl.advance(Q_block_ptr, offsets=($(BLOCK_M), 0))
    m_ptrs = m_ptrs + $(BLOCK_M)
  }
  o_offset = (off_z).to(tl.int64) * $(stride_oz) + (off_h).to(tl.int64) * $(stride_oh)
  o_range = (tl.arange(0, $(BLOCK_N))).to(tl.int64) + start_n * $(BLOCK_N)
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, (o).to(tl.float32),
    mask=o_range < $(NKV_CTX))
}


end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.TypedSource

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.TypedSource
abbrev widened := VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.TypedSource.original
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.TypedSource

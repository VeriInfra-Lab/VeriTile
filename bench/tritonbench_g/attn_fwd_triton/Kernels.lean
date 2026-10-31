import VeriTile.Triton.DSL

/-! Complete separately named repair transcription. Q/K/V and scales are FP32;
score-dot/accumulator FP32, probability and V casts/value-dot FP16, output BF16.
Dynamic padded head width, independent per-role strides, 64-bit index casts,
explicit zero masked loads, bounded diagonal loops and exact causal masks.
This is source syntax. Successful whole-host execution is not proved here. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.TypedSource
open VeriTile.Triton

def repaired
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE : Nat)
    (_hStage : STAGE = 2 ∨ STAGE = 3) : ComputeKernel :=
  let FIRST_DIAGONAL := decide (STAGE = 2)
  triton {
    tl.ports inputs=[Q,K,V,Q_scale,K_scale], outputs=[Out]
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32,
      Q_scale = tl.float32, K_scale = tl.float32, Out = tl.bfloat16
    n_ctx = ($(N_CTX)).to(tl.int64)
    stride_kn64 = ($(stride_kn)).to(tl.int64)
    stride_vk64 = ($(stride_vk)).to(tl.int64)
    start_m = (tl.program_id(0)).to(tl.int64)
    off_hz = (tl.program_id(1)).to(tl.int64)

    off_z = off_hz // $(H)
    off_h = off_hz % $(H)
    qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
    k_offset = off_z * $(stride_kz) + off_h * $(stride_kh)
    v_offset = off_z * $(stride_vz) + off_h * $(stride_vh)
    o_offset = off_z * $(stride_oz) + off_h * $(stride_oh)
    q_scale_offset = off_hz * tl.cdiv(n_ctx, $(BLOCK_M))
    k_scale_offset = off_hz * tl.cdiv(n_ctx, $(BLOCK_N))

    offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
    offs_n = (tl.arange(0, $(BLOCK_N))).to(tl.int64)
    offs_k = (tl.arange(0, $(BLOCK_DMODEL))).to(tl.int64)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + k_offset + offs_k[:, None] * $(stride_kk) + offs_n[None, :] * stride_kn64
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk64 + offs_k[None, :] * $(stride_vn)
    O_block_ptr = Out + o_offset + offs_m[:, None] * $(stride_om) + offs_k[None, :] * $(stride_on)
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    q = tl.load(Q_ptrs,
      mask=(offs_m[:, None] < n_ctx) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :], other=0.0)
    q_scale = tl.load(Q_scale_ptr)
    -- Local helper pointers: the first call does not mutate the caller's bases.
    first_lo = 0
    first_hi = start_m * $(BLOCK_M)
    K_ptrs_first = K_ptrs
    K_scale_ptr_first = K_scale_ptr
    V_ptrs_first = V_ptrs
    if FIRST_DIAGONAL {
      first_lo = start_m * $(BLOCK_M)
      first_hi = tl.minimum((start_m + $(1)) * $(BLOCK_M), n_ctx)
      first_lo = tl.multiple_of(first_lo, $(BLOCK_M))
      K_scale_ptr_first += first_lo // $(BLOCK_N)
      K_ptrs_first += stride_kn64 * first_lo
      V_ptrs_first += stride_vk64 * first_lo
    }
    for start_n in range(first_lo, first_hi, $(BLOCK_N)) {
      start_n = tl.multiple_of(start_n, $(BLOCK_N))
      k_mask = (offs_n[None, :] < (n_ctx - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[:, None]
      k = tl.load(K_ptrs_first, mask=k_mask, other=0.0)
      k_scale = tl.load(K_scale_ptr_first)
      qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
      if FIRST_DIAGONAL {
        mask_first = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask_first & (offs_n[None, :] < (n_ctx - start_n)), qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
      } else {
        qk = tl.where(offs_n[None, :] < (n_ctx - start_n), qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
      }
      p = tl.math.exp2(qk)
      l_ij = tl.sum(p, 1)
      alpha = tl.math.exp2(m_i - m_ij)
      l_i = l_i * alpha + l_ij
      acc = acc * alpha[:, None]
      v = tl.load(V_ptrs_first,
        mask=(offs_n[:, None] < (n_ctx - start_n)) &
          (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :], other=0.0)
      p = (p).to(tl.float16)
      acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
      m_i = m_ij
      K_ptrs_first += $(BLOCK_N) * stride_kn64
      K_scale_ptr_first += $(1)
      V_ptrs_first += $(BLOCK_N) * stride_vk64
    }
    -- The second helper starts from the caller's original pointers and advances
    -- to the diagonal block, exactly as its STAGE=2 prefix does.
    second_lo = start_m * $(BLOCK_M)
    second_hi = tl.minimum((start_m + $(1)) * $(BLOCK_M), n_ctx)
    second_lo = tl.multiple_of(second_lo, $(BLOCK_M))
    K_scale_ptr_second = K_scale_ptr + second_lo // $(BLOCK_N)
    K_ptrs_second = K_ptrs + stride_kn64 * second_lo
    V_ptrs_second = V_ptrs + stride_vk64 * second_lo
    for start_n in range(second_lo, second_hi, $(BLOCK_N)) {
      start_n = tl.multiple_of(start_n, $(BLOCK_N))
      k_mask = (offs_n[None, :] < (n_ctx - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[:, None]
      k = tl.load(K_ptrs_second, mask=k_mask, other=0.0)
      k_scale = tl.load(K_scale_ptr_second)
      qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
      mask = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = tl.where(mask & (offs_n[None, :] < (n_ctx - start_n)), qk, -float("inf"))
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
      p = tl.math.exp2(qk)
      l_ij = tl.sum(p, 1)
      alpha = tl.math.exp2(m_i - m_ij)
      l_i = l_i * alpha + l_ij
      acc = acc * alpha[:, None]
      v = tl.load(V_ptrs_second,
        mask=(offs_n[:, None] < (n_ctx - start_n)) &
          (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :], other=0.0)
      p = (p).to(tl.float16)
      acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
      m_i = m_ij
      K_ptrs_second += $(BLOCK_N) * stride_kn64
      K_scale_ptr_second += $(1)
      V_ptrs_second += $(BLOCK_N) * stride_vk64
    }
    acc = acc / l_i[:, None]
    -- Out.type.element_ty is bf16 in the Python wrapper.
    tl.store(O_block_ptr, (acc).to(tl.bfloat16),
      mask=(offs_m[:, None] < n_ctx) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :])
  }

end VeriTile.Bench.Repairs.AttnFwdTriton.TypedSource

namespace VeriTile.Bench.Repairs.AttnFwdTritonTileChecked
open VeriTile.Triton

/-- Integer CSE candidate sharing the quotient used by off_z. All attention
operations, stages, index casts, masks and source precision are retained. -/
def quotientRemainder
    (Q K V Q_scale K_scale Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N BLOCK_DMODEL STAGE : Nat)
    (_hStage : STAGE = 2 ∨ STAGE = 3) : ComputeKernel :=
  let FIRST_DIAGONAL := decide (STAGE = 2)
  triton {
    tl.ports inputs=[Q,K,V,Q_scale,K_scale], outputs=[Out]
    tl.region Q = tl.float32, K = tl.float32, V = tl.float32,
      Q_scale = tl.float32, K_scale = tl.float32, Out = tl.bfloat16
    n_ctx = ($(N_CTX)).to(tl.int64)
    stride_kn64 = ($(stride_kn)).to(tl.int64)
    stride_vk64 = ($(stride_vk)).to(tl.int64)
    start_m = (tl.program_id(0)).to(tl.int64)
    off_hz = (tl.program_id(1)).to(tl.int64)

    off_z = off_hz // $(H)
    off_h = off_hz - (off_hz // $(H)) * $(H)
    qvk_offset = (off_z).to(tl.int64) * $(stride_qz) + (off_h).to(tl.int64) * $(stride_qh)
    k_offset = off_z * $(stride_kz) + off_h * $(stride_kh)
    v_offset = off_z * $(stride_vz) + off_h * $(stride_vh)
    o_offset = off_z * $(stride_oz) + off_h * $(stride_oh)
    q_scale_offset = off_hz * tl.cdiv(n_ctx, $(BLOCK_M))
    k_scale_offset = off_hz * tl.cdiv(n_ctx, $(BLOCK_N))

    offs_m = start_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
    offs_n = (tl.arange(0, $(BLOCK_N))).to(tl.int64)
    offs_k = (tl.arange(0, $(BLOCK_DMODEL))).to(tl.int64)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * $(stride_qm) + offs_k[None, :] * $(stride_qk)
    Q_scale_ptr = Q_scale + q_scale_offset + start_m
    K_ptrs = K + k_offset + offs_k[:, None] * $(stride_kk) + offs_n[None, :] * stride_kn64
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + v_offset + offs_n[:, None] * stride_vk64 + offs_k[None, :] * $(stride_vn)
    O_block_ptr = Out + o_offset + offs_m[:, None] * $(stride_om) + offs_k[None, :] * $(stride_on)
    m_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([$(BLOCK_M)], dtype=tl.float32) + 1.0
    acc = tl.zeros([$(BLOCK_M), $(BLOCK_DMODEL)], dtype=tl.float32)
    q = tl.load(Q_ptrs,
      mask=(offs_m[:, None] < n_ctx) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :], other=0.0)
    q_scale = tl.load(Q_scale_ptr)
    -- Local helper pointers: the first call does not mutate the caller's bases.
    first_lo = 0
    first_hi = start_m * $(BLOCK_M)
    K_ptrs_first = K_ptrs
    K_scale_ptr_first = K_scale_ptr
    V_ptrs_first = V_ptrs
    if FIRST_DIAGONAL {
      first_lo = start_m * $(BLOCK_M)
      first_hi = tl.minimum((start_m + $(1)) * $(BLOCK_M), n_ctx)
      first_lo = tl.multiple_of(first_lo, $(BLOCK_M))
      K_scale_ptr_first += first_lo // $(BLOCK_N)
      K_ptrs_first += stride_kn64 * first_lo
      V_ptrs_first += stride_vk64 * first_lo
    }
    for start_n in range(first_lo, first_hi, $(BLOCK_N)) {
      start_n = tl.multiple_of(start_n, $(BLOCK_N))
      k_mask = (offs_n[None, :] < (n_ctx - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[:, None]
      k = tl.load(K_ptrs_first, mask=k_mask, other=0.0)
      k_scale = tl.load(K_scale_ptr_first)
      qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
      if FIRST_DIAGONAL {
        mask_first = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(mask_first & (offs_n[None, :] < (n_ctx - start_n)), qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
      } else {
        qk = tl.where(offs_n[None, :] < (n_ctx - start_n), qk, -float("inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
      }
      p = tl.math.exp2(qk)
      l_ij = tl.sum(p, 1)
      alpha = tl.math.exp2(m_i - m_ij)
      l_i = l_i * alpha + l_ij
      acc = acc * alpha[:, None]
      v = tl.load(V_ptrs_first,
        mask=(offs_n[:, None] < (n_ctx - start_n)) &
          (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :], other=0.0)
      p = (p).to(tl.float16)
      acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
      m_i = m_ij
      K_ptrs_first += $(BLOCK_N) * stride_kn64
      K_scale_ptr_first += $(1)
      V_ptrs_first += $(BLOCK_N) * stride_vk64
    }
    -- The second helper starts from the caller's original pointers and advances
    -- to the diagonal block, exactly as its STAGE=2 prefix does.
    second_lo = start_m * $(BLOCK_M)
    second_hi = tl.minimum((start_m + $(1)) * $(BLOCK_M), n_ctx)
    second_lo = tl.multiple_of(second_lo, $(BLOCK_M))
    K_scale_ptr_second = K_scale_ptr + second_lo // $(BLOCK_N)
    K_ptrs_second = K_ptrs + stride_kn64 * second_lo
    V_ptrs_second = V_ptrs + stride_vk64 * second_lo
    for start_n in range(second_lo, second_hi, $(BLOCK_N)) {
      start_n = tl.multiple_of(start_n, $(BLOCK_N))
      k_mask = (offs_n[None, :] < (n_ctx - start_n)) &
        (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[:, None]
      k = tl.load(K_ptrs_second, mask=k_mask, other=0.0)
      k_scale = tl.load(K_scale_ptr_second)
      qk = (tl.dot(q, k)).to(tl.float32) * q_scale * k_scale
      mask = offs_m[:, None] >= (start_n + offs_n[None, :])
      qk = tl.where(mask & (offs_n[None, :] < (n_ctx - start_n)), qk, -float("inf"))
      m_ij = tl.maximum(m_i, tl.max(qk, 1))
      qk -= m_ij[:, None]
      p = tl.math.exp2(qk)
      l_ij = tl.sum(p, 1)
      alpha = tl.math.exp2(m_i - m_ij)
      l_i = l_i * alpha + l_ij
      acc = acc * alpha[:, None]
      v = tl.load(V_ptrs_second,
        mask=(offs_n[:, None] < (n_ctx - start_n)) &
          (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :], other=0.0)
      p = (p).to(tl.float16)
      acc += tl.dot(p, (v).to(tl.float16), out_dtype=tl.float16)
      m_i = m_ij
      K_ptrs_second += $(BLOCK_N) * stride_kn64
      K_scale_ptr_second += $(1)
      V_ptrs_second += $(BLOCK_N) * stride_vk64
    }
    acc = acc / l_i[:, None]
    -- Out.type.element_ty is bf16 in the Python wrapper.
    tl.store(O_block_ptr, (acc).to(tl.bfloat16),
      mask=(offs_m[:, None] < n_ctx) & (tl.arange(0, $(BLOCK_DMODEL)) < $(HEAD_DIM))[None, :])
  }

end VeriTile.Bench.Repairs.AttnFwdTritonTileChecked

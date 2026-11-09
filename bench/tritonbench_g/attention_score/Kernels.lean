import VeriTile.Triton.DSL.KeywordBlockAdvance
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.TypedSource
open VeriTile.Triton
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false
def callSignature : CallSignature where
  pointers := [("Q",.fp32),("K",.fp32),("M",.fp32),("Out",.fp32)]
  scalars := [("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_on",.integer),("Z",.integer),("H",.integer),("H_KV",.integer),("N_CTX",.integer),("ROUND_CTX",.integer),("NKV_CTX",.integer),("sliding_window_offset",.integer),("sliding_window_size",.integer),("sm_scale",.fp32)]
  constexpr := ["SLIDING_WINDOW","COMPLEMENT_SLIDING_WINDOW","IS_EVEN_M","IS_EVEN_N","BLOCK_M","BLOCK_DMODEL","BLOCK_N"]
  readPorts := ["Q","K","M"]
  writePorts := ["Out"]

/-- Same typed FP32 argument expression as `ScalarArgument.fp32`, written
with constructors so this source AST remains executable Lean data. -/
def scaleArgument (value : ℝ) : ScalarArgument :=
  ⟨"sm_scale",.real,.compute (.alg .fp32 (.const value))⟩

def originalKernel
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel :=
  with_triton_call callSignature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_oz := stride_oz, stride_oh := stride_oh, stride_on := stride_on, Z := Z, H := H, H_KV := H_KV, N_CTX := N_CTX, ROUND_CTX := ROUND_CTX, NKV_CTX := NKV_CTX, sliding_window_offset := sliding_window_offset, sliding_window_size := sliding_window_size] in triton {
  start_n = (tl.program_id(0)).to(tl.int64)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // H
  off_h = off_hz % H
  off_hkv = off_h // (H//H_KV)
  q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
  k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
  m_ptrs = M + off_hz * ROUND_CTX + tl.arange(0, BLOCK_M).to(tl.int64)
  o = tl.zeros([BLOCK_M], dtype=tl.float32)

  Q_block_ptr = tl.make_block_ptr(
      base=Q + q_offset,
      shape=(N_CTX, BLOCK_DMODEL),
      strides=(stride_qm, stride_qk),
      offsets=(0, 0),
      block_shape=(BLOCK_M, BLOCK_DMODEL),
      order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
      base=K + k_offset,
      shape=(BLOCK_DMODEL, NKV_CTX),
      strides=(stride_kk, stride_kn),
      offsets=(0, (start_n * BLOCK_N).to(tl.int32)),
      block_shape=(BLOCK_DMODEL, BLOCK_N),
      order=(0, 1),
  )

  if IS_EVEN_N:
      k = tl.load(K_block_ptr)
  else:
      k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")


  lo = 0
  hi = tl.cast(N_CTX, tl.int64)
  qk_scale = sm_scale
  qk_scale *= 1.4426950408889634   # 1/log(2)

  for start_m in range(lo, hi, BLOCK_M):
      start_m = tl.multiple_of(start_m, BLOCK_M)
      if IS_EVEN_M:
          q = tl.load(Q_block_ptr)
      else:
          q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

      m = tl.load(m_ptrs, mask=start_m + tl.arange(0, BLOCK_M) < N_CTX, other=0)

      # calc qk
      qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      qk += tl.dot(q, k)
      qk = qk * qk_scale

      if SLIDING_WINDOW:
          dist = (tl.arange(0, BLOCK_M)[:, None]).to(tl.int64) - (tl.arange(0, BLOCK_N)[None, :]).to(tl.int64) \
               + start_m - start_n * BLOCK_N + tl.cast(sliding_window_offset, tl.int64)

          if COMPLEMENT_SLIDING_WINDOW:
              mask = (dist >= tl.cast(sliding_window_size, tl.int64))
          else:
              mask = (dist >= 0) & (dist < tl.cast(sliding_window_size, tl.int64))

      qk = qk - m[:, None]
      p = tl.math.exp2(qk) # (BLOCK_M, BLOCK_N)

      if SLIDING_WINDOW:
          p = tl.where(mask, p, 0)

      if not IS_EVEN_M:
          p = tl.where(
              ((tl.arange(0, BLOCK_M) + start_m) < N_CTX)[:, None],
              p, 0
          )

      o += tl.sum(p, axis=0)


      Q_block_ptr = tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))
      m_ptrs = m_ptrs + BLOCK_M

  o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
  o_range = tl.arange(0, BLOCK_N).to(tl.int64) + start_n * BLOCK_N # orange
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, o.to(Out.type.element_ty), mask = o_range < NKV_CTX)
}

/-- Bind the FP32 runtime scalar before entering the pasted function body. -/
def original
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel :=
  (originalKernel Q K M Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX BLOCK_M BLOCK_DMODEL BLOCK_N sliding_window_offset sliding_window_size sm_scale SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N _hBlockMN).withScalarArguments [scaleArgument sm_scale]
/-- Existing scalar-left FP32 score multiplication candidate. -/
def scalarLeftKernel
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel :=
  with_triton_call callSignature call_args [stride_qz := stride_qz, stride_qh := stride_qh, stride_qm := stride_qm, stride_qk := stride_qk, stride_kz := stride_kz, stride_kh := stride_kh, stride_kn := stride_kn, stride_kk := stride_kk, stride_oz := stride_oz, stride_oh := stride_oh, stride_on := stride_on, Z := Z, H := H, H_KV := H_KV, N_CTX := N_CTX, ROUND_CTX := ROUND_CTX, NKV_CTX := NKV_CTX, sliding_window_offset := sliding_window_offset, sliding_window_size := sliding_window_size] in triton {
  start_n = (tl.program_id(0)).to(tl.int64)
  off_hz = (tl.program_id(1)).to(tl.int64)
  off_z = off_hz // H
  off_h = off_hz % H
  off_hkv = off_h // (H//H_KV)
  q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
  k_offset = off_z.to(tl.int64) * stride_kz + off_hkv.to(tl.int64) * stride_kh
  m_ptrs = M + off_hz * ROUND_CTX + tl.arange(0, BLOCK_M).to(tl.int64)
  o = tl.zeros([BLOCK_M], dtype=tl.float32)

  Q_block_ptr = tl.make_block_ptr(
      base=Q + q_offset,
      shape=(N_CTX, BLOCK_DMODEL),
      strides=(stride_qm, stride_qk),
      offsets=(0, 0),
      block_shape=(BLOCK_M, BLOCK_DMODEL),
      order=(1, 0),
  )
  K_block_ptr = tl.make_block_ptr(
      base=K + k_offset,
      shape=(BLOCK_DMODEL, NKV_CTX),
      strides=(stride_kk, stride_kn),
      offsets=(0, (start_n * BLOCK_N).to(tl.int32)),
      block_shape=(BLOCK_DMODEL, BLOCK_N),
      order=(0, 1),
  )

  if IS_EVEN_N:
      k = tl.load(K_block_ptr)
  else:
      k = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")


  lo = 0
  hi = tl.cast(N_CTX, tl.int64)
  qk_scale = sm_scale
  qk_scale *= 1.4426950408889634   # 1/log(2)

  for start_m in range(lo, hi, BLOCK_M):
      start_m = tl.multiple_of(start_m, BLOCK_M)
      if IS_EVEN_M:
          q = tl.load(Q_block_ptr)
      else:
          q = tl.load(Q_block_ptr, boundary_check=(0,1), padding_option="zero")

      m = tl.load(m_ptrs, mask=start_m + tl.arange(0, BLOCK_M) < N_CTX, other=0)

      # calc qk
      qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      qk += tl.dot(q, k)
      qk = qk_scale * qk

      if SLIDING_WINDOW:
          dist = (tl.arange(0, BLOCK_M)[:, None]).to(tl.int64) - (tl.arange(0, BLOCK_N)[None, :]).to(tl.int64) \
               + start_m - start_n * BLOCK_N + tl.cast(sliding_window_offset, tl.int64)

          if COMPLEMENT_SLIDING_WINDOW:
              mask = (dist >= tl.cast(sliding_window_size, tl.int64))
          else:
              mask = (dist >= 0) & (dist < tl.cast(sliding_window_size, tl.int64))

      qk = qk - m[:, None]
      p = tl.math.exp2(qk) # (BLOCK_M, BLOCK_N)

      if SLIDING_WINDOW:
          p = tl.where(mask, p, 0)

      if not IS_EVEN_M:
          p = tl.where(
              ((tl.arange(0, BLOCK_M) + start_m) < N_CTX)[:, None],
              p, 0
          )

      o += tl.sum(p, axis=0)


      Q_block_ptr = tl.advance(Q_block_ptr, offsets=(BLOCK_M, 0))
      m_ptrs = m_ptrs + BLOCK_M

  o_offset = off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh
  o_range = tl.arange(0, BLOCK_N).to(tl.int64) + start_n * BLOCK_N # orange
  o_ptrs = Out + o_offset + o_range
  tl.store(o_ptrs, o.to(Out.type.element_ty), mask = o_range < NKV_CTX)
}

/-- Bind the FP32 runtime scalar before entering the pasted function body. -/
def scalarLeft
    (Q K M Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_oz stride_oh stride_on
      Z H H_KV N_CTX ROUND_CTX NKV_CTX
      BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)
    (sliding_window_offset sliding_window_size : Int) (sm_scale : ℝ)
    (SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N : Bool)
    (_hBlockMN : BLOCK_M = BLOCK_N) :
    ComputeKernel :=
  (scalarLeftKernel Q K M Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_oz stride_oh stride_on Z H H_KV N_CTX ROUND_CTX NKV_CTX BLOCK_M BLOCK_DMODEL BLOCK_N sliding_window_offset sliding_window_size sm_scale SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW IS_EVEN_M IS_EVEN_N _hBlockMN).withScalarArguments [scaleArgument sm_scale]
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.TypedSource

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.TypedSource
abbrev widened := VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.TypedSource.original
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.Whole002.TypedSource

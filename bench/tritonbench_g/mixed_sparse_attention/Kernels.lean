import VeriTile.Triton.DSL.NonnegativeReadViews
import VeriTile.Triton.DSL.FloatDTypeConstants
import VeriTile.Triton.Float.Structural
namespace VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.TypedSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedVariables false
inductive Storage where | fp16 | bf16 deriving DecidableEq
 def Storage.float : Storage → FloatDType | .fp16=>.fp16 | .bf16=>.bf16
 def Storage.tile (s : Storage) : TileDType := s.float.toTileDType
/-! Complete repaired Python functions. FP32 tensor storage, signed32
metadata and scalar arguments are supplied by the caller. The existing checked
nonnegative metadata view is explicit outside each quotation. -/
def callSignature : CallSignature where
  pointers := [("Q",.fp32),("K",.fp32),("V",.fp32),("seqlens",.i32),("block_count",.i32),("block_offset",.i32),("column_count",.i32),("column_index",.i32),("Out",.fp32)]
  scalars := [("sm_scale",.fp32),("stride_qz",.integer),("stride_qh",.integer),("stride_qm",.integer),("stride_qk",.integer),("stride_kz",.integer),("stride_kh",.integer),("stride_kn",.integer),("stride_kk",.integer),("stride_vz",.integer),("stride_vh",.integer),("stride_vn",.integer),("stride_vk",.integer),("stride_oz",.integer),("stride_oh",.integer),("stride_om",.integer),("stride_ok",.integer),("Z",.integer),("H",.integer),("N_CTX",.integer),("NUM_ROWS",.integer),("NNZ_S",.integer),("NNZ_V",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","BLOCK_DMODEL","dtype"]
  readPorts := []
  writePorts := []

def FP16.sourceKernel
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
    ComputeKernel :=
  let dtype : FloatDType := .fp16
  with_triton_natural_reads [seqlens,block_count,block_offset,column_count,column_index] in
  with_triton_call callSignature call_args [sm_scale := sm_scale,stride_qz := stride_qz,stride_qh := stride_qh,stride_qm := stride_qm,stride_qk := stride_qk,stride_kz := stride_kz,stride_kh := stride_kh,stride_kn := stride_kn,stride_kk := stride_kk,stride_vz := stride_vz,stride_vh := stride_vh,stride_vn := stride_vn,stride_vk := stride_vk,stride_oz := stride_oz,stride_oh := stride_oh,stride_om := stride_om,stride_ok := stride_ok,Z := Z,H := H,N_CTX := N_CTX,NUM_ROWS := NUM_ROWS,NNZ_S := NNZ_S,NNZ_V := NNZ_V] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    # Traverse every count-selected block.
    max_num_blks = num_blks
    for block_index in range(max_num_blks):
        # Use a conditional mask instead of break.
        cond = block_index < num_blks
        # Perform the following operations only when cond is True.
        start_n = tl.load(blks_ptr + block_index, mask=cond, other=0)
        cols = start_n + offs_n
        n_mask = (cols < seqlen) & cond
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = (cols[None, :] <= offs_m[:, None])
        qk = tl.where(m_mask & n_mask[None, :] & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Traverse every count-selected column.
    max_num_cols = num_cols
    for start_n in range(0, max_num_cols, BLOCK_N):
        # Use a conditional mask instead of break.
        cond = start_n < num_cols
        n_mask = (start_n + offs_n < num_cols) & cond
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = tl.where(l_i[:, None] == 0, 0.0, acc / l_i[:, None])
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
  }

def FP16.fusedMaskKernel
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
    ComputeKernel :=
  let dtype : FloatDType := .fp16
  with_triton_natural_reads [seqlens,block_count,block_offset,column_count,column_index] in
  with_triton_call callSignature call_args [sm_scale := sm_scale,stride_qz := stride_qz,stride_qh := stride_qh,stride_qm := stride_qm,stride_qk := stride_qk,stride_kz := stride_kz,stride_kh := stride_kh,stride_kn := stride_kn,stride_kk := stride_kk,stride_vz := stride_vz,stride_vh := stride_vh,stride_vn := stride_vn,stride_vk := stride_vk,stride_oz := stride_oz,stride_oh := stride_oh,stride_om := stride_om,stride_ok := stride_ok,Z := Z,H := H,N_CTX := N_CTX,NUM_ROWS := NUM_ROWS,NNZ_S := NNZ_S,NNZ_V := NNZ_V] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    # Traverse every count-selected block.
    max_num_blks = num_blks
    for block_index in range(max_num_blks):
        # Use a conditional mask instead of break.
        cond = block_index < num_blks
        # Perform the following operations only when cond is True.
        start_n = tl.load(blks_ptr + block_index, mask=cond, other=0)
        cols = start_n + offs_n
        n_mask = (cols < seqlen) & cond
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = (cols[None, :] <= offs_m[:, None])
        qk = tl.where(m_mask & n_mask[None, :] & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Traverse every count-selected column.
    max_num_cols = num_cols
    for start_n in range(0, max_num_cols, BLOCK_N):
        # Use a conditional mask instead of break.
        cond = start_n < num_cols
        n_mask = (start_n + offs_n < num_cols) & cond
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.where(m_mask & n_mask, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = tl.where(l_i[:, None] == 0, 0.0, acc / l_i[:, None])
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
  }

def BF16.sourceKernel
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
    ComputeKernel :=
  let dtype : FloatDType := .bf16
  with_triton_natural_reads [seqlens,block_count,block_offset,column_count,column_index] in
  with_triton_call callSignature call_args [sm_scale := sm_scale,stride_qz := stride_qz,stride_qh := stride_qh,stride_qm := stride_qm,stride_qk := stride_qk,stride_kz := stride_kz,stride_kh := stride_kh,stride_kn := stride_kn,stride_kk := stride_kk,stride_vz := stride_vz,stride_vh := stride_vh,stride_vn := stride_vn,stride_vk := stride_vk,stride_oz := stride_oz,stride_oh := stride_oh,stride_om := stride_om,stride_ok := stride_ok,Z := Z,H := H,N_CTX := N_CTX,NUM_ROWS := NUM_ROWS,NNZ_S := NNZ_S,NNZ_V := NNZ_V] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    # Traverse every count-selected block.
    max_num_blks = num_blks
    for block_index in range(max_num_blks):
        # Use a conditional mask instead of break.
        cond = block_index < num_blks
        # Perform the following operations only when cond is True.
        start_n = tl.load(blks_ptr + block_index, mask=cond, other=0)
        cols = start_n + offs_n
        n_mask = (cols < seqlen) & cond
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = (cols[None, :] <= offs_m[:, None])
        qk = tl.where(m_mask & n_mask[None, :] & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Traverse every count-selected column.
    max_num_cols = num_cols
    for start_n in range(0, max_num_cols, BLOCK_N):
        # Use a conditional mask instead of break.
        cond = start_n < num_cols
        n_mask = (start_n + offs_n < num_cols) & cond
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = tl.where(l_i[:, None] == 0, 0.0, acc / l_i[:, None])
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
  }

def BF16.fusedMaskKernel
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
    ComputeKernel :=
  let dtype : FloatDType := .bf16
  with_triton_natural_reads [seqlens,block_count,block_offset,column_count,column_index] in
  with_triton_call callSignature call_args [sm_scale := sm_scale,stride_qz := stride_qz,stride_qh := stride_qh,stride_qm := stride_qm,stride_qk := stride_qk,stride_kz := stride_kz,stride_kh := stride_kh,stride_kn := stride_kn,stride_kk := stride_kk,stride_vz := stride_vz,stride_vh := stride_vh,stride_vn := stride_vn,stride_vk := stride_vk,stride_oz := stride_oz,stride_oh := stride_oh,stride_om := stride_om,stride_ok := stride_ok,Z := Z,H := H,N_CTX := N_CTX,NUM_ROWS := NUM_ROWS,NNZ_S := NNZ_S,NNZ_V := NNZ_V] in triton {
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
    q = (q * qk_scale).to(dtype)

    m_mask = offs_m[:, None] < seqlen

    # Traverse every count-selected block.
    max_num_blks = num_blks
    for block_index in range(max_num_blks):
        # Use a conditional mask instead of break.
        cond = block_index < num_blks
        # Perform the following operations only when cond is True.
        start_n = tl.load(blks_ptr + block_index, mask=cond, other=0)
        cols = start_n + offs_n
        n_mask = (cols < seqlen) & cond
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = (cols[None, :] <= offs_m[:, None])
        qk = tl.where(m_mask & n_mask[None, :] & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # Traverse every count-selected column.
    max_num_cols = num_cols
    for start_n in range(0, max_num_cols, BLOCK_N):
        # Use a conditional mask instead of break.
        cond = start_n < num_cols
        n_mask = (start_n + offs_n < num_cols) & cond
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)
        qk = tl.where(m_mask & n_mask, tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32), float("-inf"))
        qk += tl.dot(q, k)
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        m_safe = tl.where(m_i_new == float("-inf"), 0.0, m_i_new)
        alpha = tl.math.exp2(m_i - m_safe)
        p = tl.math.exp2(qk - m_safe[:, None])
        acc_scale = l_i * 0 + alpha
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    acc = tl.where(l_i[:, None] == 0, 0.0, acc / l_i[:, None])
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)
  }

end
end VeriTile.Bench.Repairs.MixedSparseAttention.Whole002.TypedSource

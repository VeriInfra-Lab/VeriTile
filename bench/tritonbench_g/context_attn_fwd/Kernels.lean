import bench.tritonbench_g.context_attn_fwd.Proofs.Sources.DTypes
import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.IntegerWhereLiterals

/-! Complete repaired Python function bodies; invocation types and scalar
bindings stay outside the quotation. Every source cast and mask is retained. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnFwd
noncomputable section
def signature (dtype : Storage) : CallSignature where
  pointers := [("Q",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("K",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("V",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("Out",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("B_Start_Loc",.i32),("B_Seqlen",.i32),("b_prompt_cache_len",.i32)]
  scalars := [("sm_scale",.fp32),("stride_qbs",.integer),("stride_qh",.integer),("stride_qd",.integer),("stride_kb",.integer),("stride_kh",.integer),("stride_ks",.integer),("stride_kd",.integer),("stride_vb",.integer),("stride_vh",.integer),("stride_vs",.integer),("stride_vd",.integer),("stride_obs",.integer),("stride_oh",.integer),("stride_od",.integer),("kv_group_num",.integer)]
  constexpr := ["H","BLOCK_DMODEL","BLOCK_M","BLOCK_N"]
  readPorts := ["Q","K","V","B_Start_Loc","B_Seqlen","b_prompt_cache_len"]
  writePorts := ["Out"]

def originalKernel (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [sm_scale := sm_scale, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kb := stride_kb, stride_kh := stride_kh, stride_ks := stride_ks, stride_kd := stride_kd, stride_vb := stride_vb, stride_vh := stride_vh, stride_vs := stride_vs, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num] in triton {
    start_m = tl.program_id(0).to(tl.int64)
    cur_bh = tl.program_id(1).to(tl.int64)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64) - prompt_cache_len
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        off_k = cur_batch * stride_kb + (start_n + offs_n[None, :]) * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=start_n + offs_n[None, :] < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :]
        qk = tl.where(mask, qk * sm_scale, -100000000.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = cur_batch * stride_vb + (start_n + offs_n[:, None]) * stride_vs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=start_n + offs_n[:, None] < block_end_loc, other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
  }

def scaleFirstKernel (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [sm_scale := sm_scale, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kb := stride_kb, stride_kh := stride_kh, stride_ks := stride_ks, stride_kd := stride_kd, stride_vb := stride_vb, stride_vh := stride_vh, stride_vs := stride_vs, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num] in triton {
    start_m = tl.program_id(0).to(tl.int64)
    cur_bh = tl.program_id(1).to(tl.int64)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64) - prompt_cache_len
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        off_k = cur_batch * stride_kb + (start_n + offs_n[None, :]) * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=start_n + offs_n[None, :] < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :]
        qk = tl.where(mask, sm_scale * qk, -100000000.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = cur_batch * stride_vb + (start_n + offs_n[:, None]) * stride_vs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=start_n + offs_n[:, None] < block_end_loc, other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
  }

def blockEndFactoredKernel (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [sm_scale := sm_scale, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kb := stride_kb, stride_kh := stride_kh, stride_ks := stride_ks, stride_kd := stride_kd, stride_vb := stride_vb, stride_vh := stride_vh, stride_vs := stride_vs, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num] in triton {
    start_m = tl.program_id(0).to(tl.int64)
    cur_bh = tl.program_id(1).to(tl.int64)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64) - prompt_cache_len
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M, cur_batch_seq_len) + prompt_cache_len
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        off_k = cur_batch * stride_kb + (start_n + offs_n[None, :]) * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=start_n + offs_n[None, :] < block_end_loc, other=0.0)
        qk = tl.dot(q, k)
        mask = offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :]
        qk = tl.where(mask, qk * sm_scale, -100000000.0)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        off_v = cur_batch * stride_vb + (start_n + offs_n[:, None]) * stride_vs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=start_n + offs_n[:, None] < block_end_loc, other=0.0)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
  }

end
end VeriTile.Bench.Repairs.ContextAttnFwd

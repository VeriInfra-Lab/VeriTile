import bench.tritonbench_g.context_attn_mistral.Proofs.Sources.DTypes
import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.IntegerWhereLiterals

/-! Complete pasted repaired Python bodies. Storage and call bindings stay
outside the quotations; no casts, masks, branches or return are removed. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
def signature (dtype : Storage) : CallSignature where
  pointers := [("Q",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("K",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("V",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32),("B_Start_Loc",.i32),("B_Seqlen",.i32),("Out",match dtype with | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 => .fp32)]
  scalars := [("sm_scale",.fp32),("stride_qbs",.integer),("stride_qh",.integer),("stride_qd",.integer),("stride_kbs",.integer),("stride_kh",.integer),("stride_kd",.integer),("stride_vbs",.integer),("stride_vh",.integer),("stride_vd",.integer),("stride_obs",.integer),("stride_oh",.integer),("stride_od",.integer),("kv_group_num",.integer),("sliding_window",.integer)]
  constexpr := ["BLOCK_M","BLOCK_DMODEL","BLOCK_N"]
  readPorts := ["Q","K","V","B_Start_Loc","B_Seqlen"]
  writePorts := ["Out"]

def originalKernel (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen Out : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num : Nat) (sliding_window : Int)
    (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [sm_scale := sm_scale, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kbs := stride_kbs, stride_kh := stride_kh, stride_kd := stride_kd, stride_vbs := stride_vbs, stride_vh := stride_vh, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num, sliding_window := sliding_window] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    start_m = tl.program_id(2).to(tl.int64)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=start_n + offs_n[None, :] < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, -1000000000.0)
        qk = tl.where(offs_m[:, None] - (start_n + offs_n[None, :]) < sliding_window, qk, -1000000000.0)
        m_ij = tl.max(qk, 1)
        m_ij = tl.where(m_ij == -1000000000.0, 0.0, m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        l_i_new = tl.where(l_i_new == 0.0, 1e-09, l_i_new)
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return
  }

def blockStartReusedKernel (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen Out : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num : Nat) (sliding_window : Int)
    (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [sm_scale := sm_scale, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kbs := stride_kbs, stride_kh := stride_kh, stride_kd := stride_kd, stride_vbs := stride_vbs, stride_vh := stride_vh, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num, sliding_window := sliding_window] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    start_m = tl.program_id(2).to(tl.int64)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=start_n + offs_n[None, :] < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, -1000000000.0)
        qk = tl.where(offs_m[:, None] - (start_n + offs_n[None, :]) < sliding_window, qk, -1000000000.0)
        m_ij = tl.max(qk, 1)
        m_ij = tl.where(m_ij == -1000000000.0, 0.0, m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        l_i_new = tl.where(l_i_new == 0.0, 1e-09, l_i_new)
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return
  }

def scaleFirstKernel (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen Out : RegionName)
    (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num : Nat) (sliding_window : Int)
    (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (signature dtype) call_args [sm_scale := sm_scale, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kbs := stride_kbs, stride_kh := stride_kh, stride_kd := stride_kd, stride_vbs := stride_vbs, stride_vh := stride_vh, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num, sliding_window := sliding_window] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    start_m = tl.program_id(2).to(tl.int64)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs, mask=start_n + offs_n[None, :] < cur_batch_seq_len, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk = sm_scale * qk
        qk = tl.where(offs_m[:, None] >= start_n + offs_n[None, :], qk, -1000000000.0)
        qk = tl.where(offs_m[:, None] - (start_n + offs_n[None, :]) < sliding_window, qk, -1000000000.0)
        m_ij = tl.max(qk, 1)
        m_ij = tl.where(m_ij == -1000000000.0, 0.0, m_ij)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        l_i_new = tl.where(l_i_new == 0.0, 1e-09, l_i_new)
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs, mask=start_n + offs_n[:, None] < cur_batch_seq_len, other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)
    return
  }

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel

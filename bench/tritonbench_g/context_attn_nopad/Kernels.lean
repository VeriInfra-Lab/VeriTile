import bench.tritonbench_g.context_attn_nopad.Sources.Mathematical_HostConfig

/-! Complete selected FP32 source in the accepted bounded machine-index
projection. Its signed-width transport is unchanged. -/
namespace VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly
open VeriTile.Triton
abbrev Call := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Host.Call
abbrev Layout := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Formula.Layout
abbrev Regions := VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program.Regions

/-- Direct binding to the shared selected source. -/
def originalFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ContextAttnNopad.Program.kernel p r .fp32 Bool.false scale


def blockStartReusedFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  let Q := r.query
  let K := r.key
  let V := r.value
  let Out := r.out
  let B_Start_Loc := r.start
  let B_Seqlen := r.sequence
  triton {
  tl.ports inputs=[Q,K,V,B_Start_Loc,B_Seqlen], outputs=[Out]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32

  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_m = tl.program_id(2)

  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

  block_start_loc = $(p.BM) * start_m

  offs_n = tl.arange(0, $(p.BN))
  offs_d = tl.arange(0, $(p.D))
  offs_m = block_start_loc + tl.arange(0, $(p.BM))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(p.qRow) +
    cur_head * $(p.qHead) + offs_d[None, :] * $(p.qDim)
  off_k = offs_n[None, :] * $(p.kRow) + cur_head * $(p.kHead) +
    offs_d[:, None] * $(p.kDim)
  off_v = offs_n[:, None] * $(p.vRow) + cur_head * $(p.vHead) +
    offs_d[None, :] * $(p.vDim)

  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  k_ptrs = K + off_k
  v_ptrs = V + off_v

  m_i = tl.zeros([$(p.BM)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(p.BM)], dtype=tl.float32)
  acc = tl.zeros([$(p.BM), $(p.D)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))

  for start_n in range($((0 : Int)), block_mask * (start_m + $(1)) * $(p.BM), $(p.BN)) {
    start_n = tl.multiple_of(start_n, $(p.BN))
    k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * $(p.kRow),
      mask=(start_n + offs_n[None, :]) < cur_batch_seq_len,
      other=0.0)

    qk = tl.zeros([$(p.BM), $(p.BN)], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk *= $((scale : ℝ))
    qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))

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
    acc = acc * acc_scale[:, None]
    v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * $(p.vRow),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len,
      other=0.0)

    p = (p).to(tl.float32)
    acc += tl.dot(p, v)
    l_i = l_i_new
    m_i = m_i_new
  }
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(p.outRow) +
    cur_head * $(p.outHead) + offs_d[None, :] * $(p.outDim)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float32), mask=offs_m[:, None] < cur_batch_seq_len)
}

noncomputable def kernel (c : Call) (r : Regions) : ComputeKernel := originalFP32 c.layout r c.scale
end VeriTile.Bench.Repairs.ContextAttnNopad.KernelOnly

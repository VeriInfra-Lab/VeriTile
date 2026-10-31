import bench.tritonbench_g.context_attn_fwd.Sources.Mathematical_Program

/-! Complete selected FP32 source in the existing bounded machine-index projection.
The accepted signed-width transport remains unchanged. Other storage branches
are compatibility views of the existing shared source. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd
open VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Storage
abbrev Layout := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Formula.Layout
abbrev Regions := VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.Regions

/-- Direct binding to the shared selected source. -/
def originalFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp32 Bool.false scale


def blockEndFactoredFP32 (p : Layout) (r : Regions) (scale : ℝ) : ComputeKernel :=
  let Q := r.query
  let K := r.key
  let V := r.value
  let Out := r.out
  let B_Start_Loc := r.start
  let B_Seqlen := r.sequence
  let b_prompt_cache_len := r.prompt
  triton {
  tl.ports inputs=[Q,K,V,B_Start_Loc,B_Seqlen,b_prompt_cache_len], outputs=[Out]
  tl.region Q = tl.float32, K = tl.float32, V = tl.float32, Out = tl.float32,
    B_Start_Loc = tl.int32, B_Seqlen = tl.int32, b_prompt_cache_len = tl.int32

  start_m = tl.program_id(0)
  cur_bh = tl.program_id(1)
  cur_batch = cur_bh // $(p.heads)
  cur_head = cur_bh % $(p.heads)

  cur_kv_head = cur_head // $(p.kvGroup)
  prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len

  block_start_loc = $(p.BM) * start_m

  offs_n = tl.arange(0, $(p.BN))
  offs_d = tl.arange(0, $(p.D))
  offs_m = block_start_loc + tl.arange(0, $(p.BM))
  off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * $(p.qRow) +
    cur_head * $(p.qHead) + offs_d[None, :] * $(p.qDim)
  q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

  m_i = tl.zeros([$(p.BM)], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([$(p.BM)], dtype=tl.float32)
  acc = tl.zeros([$(p.BM), $(p.D)], dtype=tl.float32)

  block_mask = tl.where(block_start_loc < cur_batch_seq_len, $(1), $(0))
  block_end_loc = tl.minimum(block_start_loc + $(p.BM), cur_batch_seq_len) + prompt_cache_len
  for start_n in range($(0), block_mask * block_end_loc, $(p.BN)) {
    start_n = tl.multiple_of(start_n, $(p.BN))
    off_k = cur_batch * $(p.kBatch) + (start_n + offs_n[None, :]) * $(p.kRow) +
      cur_kv_head * $(p.kHead) + offs_d[:, None] * $(p.kDim)
    k = tl.load(K + off_k,
      mask=(start_n + offs_n[None, :]) < block_end_loc,
      other=0.0)

    qk = tl.dot(q, k)
    mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
    qk = tl.where(mask, qk * $((scale : ℝ)), -1.0e8)
    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    qk -= m_ij[:, None]
    p = tl.math.exp2(qk)
    l_ij = tl.sum(p, 1)

    alpha = tl.math.exp2(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]
    off_v = cur_batch * $(p.vBatch) + (start_n + offs_n[:, None]) * $(p.vRow) +
      cur_kv_head * $(p.vHead) + offs_d[None, :] * $(p.vDim)
    v = tl.load(V + off_v,
      mask=(start_n + offs_n[:, None]) < block_end_loc,
      other=0.0)

    p = (p).to(tl.float32)
    acc = tl.dot(p, v, acc)
    m_i = m_ij
  }

  acc = acc / l_i[:, None]
  off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * $(p.outRow) +
    cur_head * $(p.outHead) + offs_d[None, :] * $(p.outDim)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, (acc).to(tl.float32), mask=offs_m[:, None] < cur_batch_seq_len)
}

noncomputable def kernel (p : Layout) (r : Regions) (storage : Storage) (scale : ℝ) : ComputeKernel :=
  match storage with
  | .fp32 => originalFP32 p r scale
  | .fp16 => VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp16 Bool.false scale
  | .bf16 => VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .bf16 Bool.false scale

theorem actual_source (p : Layout) (r : Regions) (scale : ℝ) :
    kernel p r .fp32 scale =
      VeriTile.Bench.TritonBenchG.ContextAttnFwd.Program.kernel p r .fp32 Bool.false scale := rfl
end VeriTile.Bench.Repairs.ContextAttnFwd

import bench.tritonbench_g.token_attn_llama2.Sources.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
abbrev source := VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source
abbrev typed := VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.typed
theorem source_unchanged : source=VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue.source := rfl

/-- Complete candidate reuses the token-block origin instead of recomputing
start_n*BLOCK_N. The 0/1 loop, Q/K FP32 arithmetic and all masks are retained. -/
def reuseTyped (dtype : FloatDType)
    (Q K : RegionName) (sm_scale : ℝ) (B_Loc B_Start_Loc B_Seqlen : Region .nat)
    (Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num
      BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  tl.region Q = DTYPE, K = DTYPE, Att_Out = DTYPE
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  start_n = tl.program_id(2)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_start_index = $(max_input_len) - cur_batch_seq_len
  cur_batch_end_index = $(max_input_len)
  off_q = cur_batch * $(stride_qbs) + cur_head * $(stride_qh) + offs_d * $(stride_qd)
  block_stard_index = start_n * $(BLOCK_N)
  offs_n = block_stard_index + tl.arange(0, $(BLOCK_N))
  block_mask = tl.where(block_stard_index < cur_batch_seq_len, $(1), $(0))
  for start_mark in range($(0), block_mask, $(1)) {
    q = tl.load(Q + off_q + start_mark).to(tl.float32)
    offs_n_new = cur_batch_start_index + offs_n
    k_loc = tl.load(B_Loc + $(stride_b_loc_b) * cur_batch +
      $(stride_b_loc_s) * offs_n_new,
      mask=offs_n_new < cur_batch_end_index, other=$(0))
    off_k = k_loc[:, None] * $(stride_kbs) + cur_kv_head * $(stride_kh) +
      offs_d[None, :] * $(stride_kd)
    k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0).to(tl.float32)
    att_value = tl.sum(q[None, :] * k, 1)
    att_value *= $((sm_scale : ℝ))
    off_o = cur_head * $(att_stride_h) +
      (cur_batch_in_all_start_index + offs_n) * $(att_stride_bs)
    tl.store(Att_Out + off_o, att_value.to(DTYPE), mask=offs_n_new < cur_batch_end_index)
  }
}

def reuseSource (p : Parameters) : ComputeKernel :=
  reuseTyped .fp32 p.q p.k p.scale p.locations p.starts p.lengths p.output p.maxLength
    p.locationBatchStride p.locationSequenceStride p.queryBatchStride p.queryHeadStride
    p.queryDimStride p.keyBatchStride p.keyHeadStride p.keyDimStride p.outputHeadStride
    p.outputSequenceStride p.groups p.dimensions p.tokens

#axiomsClean source_unchanged
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked

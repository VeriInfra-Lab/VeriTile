import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
open VeriTile.Triton
/-- The complete supplied-probability PV source at the local fp32 storage profile.
Only storage declarations are added; every source loop, mask, arithmetic node,
cast and unmasked store remains present. -/
def source (dtype : FloatDType)
    (Prob V Out : RegionName)
    (Req_to_tokens B_req_idx B_Start_Loc B_Seqlen : Region .nat)
    (stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs
      stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
      kv_group_num BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel := triton {
  tl.region Prob = DTYPE, V = DTYPE, Out = DTYPE
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_kv_head = cur_head // $(kv_group_num)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_start_index = 0
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
  v_loc_off = cur_batch_req_idx * $(stride_req_to_tokens_b) +
    (cur_batch_start_index + offs_n) * $(stride_req_to_tokens_s)
  p_offs = cur_head * $(stride_ph) +
    (cur_batch_in_all_start_index + offs_n) * $(stride_pbs)
  v_offs = cur_kv_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_batch_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    p_value = tl.load(Prob + p_offs + start_n,
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_loc = tl.load(Req_to_tokens + v_loc_off +
      start_n * $(stride_req_to_tokens_s),
      mask=(start_n + offs_n) < cur_batch_seq_len, other=0.0)
    v_value = tl.load(V + v_offs + v_loc[:, None] * $(stride_vbs),
      mask=(start_n + offs_n[:, None]) < cur_batch_seq_len, other=0.0)
    acc += tl.sum(p_value[:, None] * v_value, 0)
  }
  acc = (acc).to(Out.dtype.element_ty)
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc)
}

structure Parameters where
  (prob value output : RegionName)
  (tokens request start length : Region .nat)
  (requestBatchStride requestTokenStride probHeadStride probTokenStride : Nat)
  (valueBatchStride valueHeadStride valueDimStride : Nat)
  (outputBatchStride outputHeadStride outputDimStride : Nat)
  (kvGroup width block : Nat)

def kernel (p : Parameters) : ComputeKernel :=
  source .fp32 p.prob p.value p.output p.tokens p.request p.start p.length
    p.requestBatchStride p.requestTokenStride p.probHeadStride p.probTokenStride
    p.valueBatchStride p.valueHeadStride p.valueDimStride
    p.outputBatchStride p.outputHeadStride p.outputDimStride p.kvGroup p.width p.block

def prelude (p : Parameters) := (kernel p).surfaceBody.take 13
def body (p : Parameters) : List ComputeStmt :=
  let code : List ComputeStmt := (kernel p).surfaceBody
  match code[13]? with
  | some (ComputeStmt.forRangeDyn _ _ _ _ xs) => xs
  | _ => []
def postlude (p : Parameters) := (kernel p).surfaceBody.drop 14

end VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages

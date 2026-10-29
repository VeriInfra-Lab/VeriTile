import VeriTile.Triton
namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton
/-- Repaired complete online loop, with fp32 storage instantiated below.
No input arithmetic is erased. Padded page-index lanes use zero and V is
masked with other=0, so neither invalid sentinel memory nor NaN padding is read.
The QK/V .to(fp32) and terminal .to(DTYPE) materialize the original implicit
same-format register/storage boundaries; NativeGraph stages these as bit
identities for the actual all-FP32 native profile. -/
def typed (dtype : FloatDType)
    (Logics V Out : RegionName) (BLoc : Region .int)
    (BStartLoc BSeqLen : Region .nat)
    (max_input_len
      stride_logic_h stride_logic_bs
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_b_loc_b stride_b_loc_s
      BLOCK_DMODEL BLOCK_N : Nat)
    (other_kv_index : Int) :
    ComputeKernel := triton {
  tl.region Logics = DTYPE, V = DTYPE, Out = DTYPE
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  cur_batch_seq_len = tl.load(BSeqLen + cur_batch)
  cur_batch_start_loc = tl.load(BStartLoc + cur_batch)
  offs_n = tl.arange(0, $(BLOCK_N))
  offs_d = tl.arange(0, $(BLOCK_DMODEL))
  off_v = cur_head * $(stride_vh) + offs_d[None, :] * $(stride_vd)
  off_b_loc = cur_batch * $(stride_b_loc_b) +
    ($(max_input_len) - cur_batch_seq_len) * $(stride_b_loc_s)
  v_ptrs = V + off_v
  e_max = float("-inf")
  e_sum = 0.0
  acc = tl.zeros([$(BLOCK_DMODEL)], dtype=tl.float32)
  for start_n in range($(0), cur_batch_seq_len, $(BLOCK_N)) {
    start_n = tl.multiple_of(start_n, $(BLOCK_N))
    v_index = tl.load(BLoc + off_b_loc +
      (start_n + offs_n) * $(stride_b_loc_s),
      mask=(start_n + offs_n)  <  cur_batch_seq_len, other=$(Int.ofNat 0))
    qk = tl.load(Logics + cur_head * $(stride_logic_h) +
        (cur_batch_start_loc + start_n + offs_n) * $(stride_logic_bs),
      mask=start_n + offs_n  <  cur_batch_seq_len, other=float("-inf")).to(tl.float32)
    n_e_max = tl.maximum(tl.max(qk, 0), e_max)
    n_e_max = tl.maximum(n_e_max, -340282346638528859811704183484516925440.0)
    old_scale = tl.exp(e_max - n_e_max)
    p = tl.exp(qk - n_e_max)
    e_sum = e_sum * old_scale + tl.sum(p, 0)
    v = tl.load(v_ptrs + v_index[:, None] * $(stride_vbs),
      mask=(start_n + offs_n  <  cur_batch_seq_len)[:, None], other=0.0).to(tl.float32)
    acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
    e_max = n_e_max
  }
  acc = acc / e_sum
  off_o = cur_batch * $(stride_obs) + cur_head * $(stride_oh) + offs_d * $(stride_od)
  out_ptrs = Out + off_o
  tl.store(out_ptrs, acc.to(DTYPE))
}


structure Parameters where
  (logics values output : RegionName)
  locations : Region .int
  (starts lengths : Region .nat)
  (maxLength logicHeadStride logicSequenceStride valueBatchStride valueHeadStride valueDimStride
    outputBatchStride outputHeadStride outputDimStride locationBatchStride locationSequenceStride
    dimensions tokens : Nat)
  sentinel : Int

def source (p : Parameters) : ComputeKernel :=
  typed .fp32 p.logics p.values p.output p.locations p.starts p.lengths p.maxLength
    p.logicHeadStride p.logicSequenceStride p.valueBatchStride p.valueHeadStride p.valueDimStride
    p.outputBatchStride p.outputHeadStride p.outputDimStride p.locationBatchStride p.locationSequenceStride
    p.dimensions p.tokens p.sentinel

end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue

import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.Typed
open VeriTile.Triton
def source (dtype : FloatDType)
    (Logics : RegionName) (B_Start_Loc B_Seqlen : Region .nat) (Prob_Out : RegionName)
    (stride_logic_h stride_logic_bs stride_prob_h stride_prob_bs BLOCK_SIZE : Nat) :
    ComputeKernel := triton {
  tl.region Logics = DTYPE, Prob_Out = DTYPE
  cur_batch = tl.program_id(0)
  cur_head = tl.program_id(1)
  col_offsets = tl.arange(0, $(BLOCK_SIZE))
  cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
  cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
  row = (tl.load(Logics + cur_head * $(stride_logic_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_logic_bs),
    mask=col_offsets < cur_batch_seq_len, other=-float("inf"))).to(tl.float32)
  row_minus_max = row - tl.max(row, axis=0)
  numerator = tl.exp(row_minus_max)
  denominator = tl.sum(numerator, axis=0)
  softmax_output = numerator / denominator
  tl.store(Prob_Out + cur_head * $(stride_prob_h) +
      (cur_batch_in_all_start_index + col_offsets) * $(stride_prob_bs),
    (softmax_output).to(DTYPE), mask=col_offsets < cur_batch_seq_len)
}

end VeriTile.Bench.TritonBenchG.TokenSoftmaxBloom.Typed

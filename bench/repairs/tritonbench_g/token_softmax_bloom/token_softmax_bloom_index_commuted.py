import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_token_softmax(Logics, B_Start_Loc, B_Seqlen, Prob_Out, stride_start: tl.int64, stride_length: tl.int64, stride_logic_h: tl.int64, stride_logic_bs: tl.int64, stride_prob_h: tl.int64, stride_prob_bs: tl.int64, BLOCK_SIZE: tl.constexpr):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + stride_length * cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch * stride_start).to(tl.int64)
    row = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_in_all_start_index + col_offsets) * stride_logic_bs, mask=col_offsets < cur_batch_seq_len, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(Prob_Out + cur_head * stride_prob_h + (cur_batch_in_all_start_index + col_offsets) * stride_prob_bs, softmax_output, mask=col_offsets < cur_batch_seq_len)

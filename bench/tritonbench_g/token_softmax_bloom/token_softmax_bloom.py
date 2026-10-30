import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_token_softmax(
    Logics, B_Start_Loc, B_Seqlen, Prob_Out,
    stride_start: tl.int64, stride_length: tl.int64,
    stride_logic_h: tl.int64, stride_logic_bs: tl.int64,
    stride_prob_h: tl.int64, stride_prob_bs: tl.int64,
    BLOCK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch * stride_length).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch * stride_start).to(tl.int64)
    row = tl.load(Logics + cur_head * stride_logic_h +
                  (cur_batch_in_all_start_index + col_offsets) * stride_logic_bs,
                  mask=col_offsets < cur_batch_seq_len, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(Prob_Out + cur_head * stride_prob_h +
             (cur_batch_in_all_start_index + col_offsets) * stride_prob_bs,
             softmax_output, mask=col_offsets < cur_batch_seq_len)


def _output_nonoverlap(rows, cols, rs, cs):
    # Either physical axis order is accepted, including sliced/offset views.
    return ((rows <= 1 or rs > (cols - 1) * cs) and (cols <= 1 or cs > 0) or
            (cols <= 1 or cs > (rows - 1) * rs) and (rows <= 1 or rs > 0))


def _check_contents(starts, lengths, batch, tokens, maximum):
    segments = []
    for b in range(batch):
        start, count = int(starts[b]), int(lengths[b])
        if start < 0 or count < 0 or count > maximum or start + count > tokens:
            raise ValueError('invalid segment bounds')
        if count:
            for previous_start, previous_count in segments:
                if not (start + count <= previous_start or
                        previous_start + previous_count <= start):
                    raise ValueError('nonempty segments must be disjoint')
            segments.append((start, count))


@torch.no_grad()
def token_softmax_fwd(Logics, B_Start_Loc, B_Seqlen, Prob_Out, max_input_len):
    """FP32 segmented softmax with checked metadata contents."""
    tensors = (Logics, B_Start_Loc, B_Seqlen, Prob_Out)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError('all four arguments must be tensors')
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError('strided tensors required')
    if Logics.dtype != torch.float32 or Prob_Out.dtype != torch.float32:
        raise TypeError('FP32 logits and output required')
    if B_Start_Loc.dtype != torch.int32 or B_Seqlen.dtype != torch.int32:
        raise TypeError('signed int32 metadata required')
    if any(not t.is_cuda for t in tensors):
        raise ValueError('CUDA tensors required')
    if any(t.device != Logics.device for t in tensors):
        raise ValueError('all tensors must share one device')
    if Logics.ndim != 2 or Prob_Out.shape != Logics.shape:
        raise ValueError('logits and output must have the same rank-two shape')
    if B_Start_Loc.ndim != 1 or B_Seqlen.shape != B_Start_Loc.shape:
        raise ValueError('metadata must have the same rank-one shape')
    if any(t.is_neg() for t in tensors):
        raise ValueError('unresolved negative views are unsupported')
    if type(max_input_len) is not int or max_input_len < 0:
        raise ValueError('max_input_len must be a nonnegative Python integer')
    batch, heads, tokens = B_Start_Loc.shape[0], Logics.shape[0], Logics.shape[1]
    if batch == 0 or heads == 0 or tokens == 0:
        return None
    if max_input_len == 0:
        _check_contents(B_Start_Loc, B_Seqlen, batch, tokens, max_input_len)
        return None
    BLOCK_SIZE = triton.next_power_of_2(max_input_len)
    if batch >= 2**31 or heads >= 2**16:
        raise ValueError('grid exceeds CUDA launch dimensions')
    if BLOCK_SIZE > 2**20:
        raise ValueError('block exceeds the Triton frontend limit')
    if not _output_nonoverlap(heads, tokens, *Prob_Out.stride()):
        raise ValueError('output strides must separate rows or columns')
    for t in tensors:
        if any(s < 0 or s >= 2**63 for s in t.stride()):
            raise ValueError('strides must fit nonnegative signed int64')
        if t.data_ptr() % 4:
            raise ValueError('four-byte data-pointer alignment required')
    formed = ((heads - 1) * Logics.stride(0) + (tokens + BLOCK_SIZE - 1) * Logics.stride(1),
              (batch - 1) * B_Start_Loc.stride(0),
              (batch - 1) * B_Seqlen.stride(0),
              (heads - 1) * Prob_Out.stride(0) + (tokens + BLOCK_SIZE - 1) * Prob_Out.stride(1))
    if tokens + BLOCK_SIZE >= 2**63 or any(n >= 2**63 for n in formed):
        raise ValueError('formed indices must fit signed int64')
    if any(t.data_ptr() + 4 * n + 4 > 2**64 for t, n in zip(tensors, formed)):
        raise ValueError('formed pointers must fit uint64')
    out_storage = Prob_Out.untyped_storage()
    out_lo, out_hi = out_storage.data_ptr(), out_storage.data_ptr() + 4 * (out_storage.nbytes() // 4)
    for t in tensors[:3]:
        storage = t.untyped_storage()
        lo, hi = storage.data_ptr(), storage.data_ptr() + 4 * (storage.nbytes() // 4)
        if not (out_hi <= lo or hi <= out_lo):
            raise ValueError('output storage must be separate from read storage')
    _check_contents(B_Start_Loc, B_Seqlen, batch, tokens, max_input_len)
    num_warps = 16 if BLOCK_SIZE >= 4096 else 8 if BLOCK_SIZE >= 2048 else 4
    with torch.cuda.device(Logics.device):
        _fwd_kernel_token_softmax[(batch, heads)](
            Logics, B_Start_Loc, B_Seqlen, Prob_Out,
            B_Start_Loc.stride(0), B_Seqlen.stride(0),
            Logics.stride(0), Logics.stride(1), Prob_Out.stride(0), Prob_Out.stride(1),
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
    return None

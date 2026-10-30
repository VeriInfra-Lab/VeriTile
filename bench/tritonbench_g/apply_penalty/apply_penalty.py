"""Checked direct, in-place FP32/int32 apply_penalty revision."""
import operator

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_apply_penalty_repaired(
    Logits, presence_penalty, freqency_penalty, repetition_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len,
    stride_logit_b, stride_logit_s, BLOCK_P: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)
    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch).to(tl.int64)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1).to(tl.int64)
    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P).to(tl.int64)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
                        mask=cur_batch_id_offset < cur_batch_end_index, other=0).to(tl.int64)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
                              mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    row_start_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64)
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64) + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)


# Existing DIV-MUL-RCP proof candidate; production still selects the original.
@triton.jit
def _fwd_kernel_apply_penalty_reciprocal(
    Logits, presence_penalty, freqency_penalty, repetition_penalty,
    p_token_ids, p_token_counts, p_cumsum_seq_len,
    stride_logit_b, stride_logit_s, BLOCK_P: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)
    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch).to(tl.int64)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1).to(tl.int64)
    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P).to(tl.int64)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
                        mask=cur_batch_id_offset < cur_batch_end_index, other=0).to(tl.int64)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
                              mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    row_start_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64)
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits * (1 / cur_repetition), cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64) + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)


def source_block(max_len):
    block = triton.next_power_of_2(max(1, max_len))
    if block <= 512:
        return 512
    if block <= 1024:
        return 1024
    return block


def _check_tensor(value, dtype, rank, device):
    if not isinstance(value, torch.Tensor):
        raise TypeError("all seven tensor arguments must be tensors")
    if value.layout != torch.strided or value.dtype != dtype or value.ndim != rank:
        raise ValueError("unsupported tensor layout, dtype, or rank")
    if value.device.type != "cuda" or value.device != device:
        raise ValueError("all tensors must be on the logits CUDA device")
    if not value.is_contiguous() or value.is_neg():
        raise ValueError("tensors must be contiguous without negative view bits")
    if value.numel() >= 2**31:
        raise ValueError("each tensor must contain fewer than 2**31 elements")
    if value.numel() and value.data_ptr() % 4:
        raise ValueError("nonempty FP32/int32 data pointers must be 4-byte aligned")


@torch.no_grad()
def apply_penalty_repaired(Logits, presence_penalty, freqency_penalty,
                           repetition_penalty, p_token_ids, p_token_counts,
                           p_cumsum_seq_len, p_max_len_in_batch):
    if not isinstance(Logits, torch.Tensor):
        raise TypeError("Logits must be a tensor")
    _check_tensor(Logits, torch.float32, 2, Logits.device)
    floating = (presence_penalty, freqency_penalty, repetition_penalty)
    integer = (p_token_ids, p_token_counts, p_cumsum_seq_len)
    for value in floating:
        _check_tensor(value, torch.float32, 1, Logits.device)
    for value in integer:
        _check_tensor(value, torch.int32, 1, Logits.device)
    max_len = operator.index(p_max_len_in_batch)
    if not 0 <= max_len <= 65536:
        raise ValueError("maximum-length hint must be between 0 and 65536")
    block = source_block(max_len)
    batches, vocab = Logits.shape
    if any(value.numel() < batches for value in floating):
        raise ValueError("penalties need a value for every row")
    if batches and p_cumsum_seq_len.numel() < batches + 1:
        raise ValueError("cumsum needs B+1 entries")
    if Logits.numel():
        lo, hi = Logits.data_ptr(), Logits.data_ptr() + 4 * Logits.numel()
        for value in floating + integer:
            if value.numel():
                start = value.data_ptr()
                if lo < start + 4 * value.numel() and start < hi:
                    raise ValueError("read-only metadata must not overlap logits")
    # These host reads validate integer metadata; the native launch still uses
    # the original seven tensors. No logits/penalty staging or copyback occurs.
    ids = p_token_ids.tolist()
    counts = p_token_counts.tolist()
    cumsum = p_cumsum_seq_len.tolist()
    for row in range(batches):
        start, stop = cumsum[row:row + 2]
        if not 0 <= start <= stop <= min(len(ids), len(counts)):
            raise ValueError("invalid cumsum window")
        if stop - start > block:
            raise ValueError("maximum-length hint leaves records outside the tile")
        seen = set()
        for pos in range(start, stop):
            token = ids[pos]
            if not 0 <= token < vocab or counts[pos] < 0:
                raise ValueError("token outside vocabulary or negative occurrence count")
            if token in seen:
                raise ValueError("token IDs must be unique within each row")
            seen.add(token)
    if batches == 0:
        return None
    with torch.cuda.device(Logits.device):
        _fwd_kernel_apply_penalty_repaired[(batches,)](
            Logits, presence_penalty, freqency_penalty, repetition_penalty,
            p_token_ids, p_token_counts, p_cumsum_seq_len, vocab, 1,
            BLOCK_P=block, num_warps=8, enable_fp_fusion=True,
        )
    return None

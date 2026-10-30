from typing import Dict, Sequence, Union
import math
import torch
import triton
import triton.language as tl


KERNEL_META = dict()

def get_kernel_meta(tensor: torch.Tensor):
    """kernel meta."""
    return KERNEL_META

TypeHintType = Union[Dict[str, type], Sequence[type], None]


@triton.jit
def _multinomial_sampling_kernel(Scores, Seeds, Offsets, Indices, Outputs,
                                 stride_sb, stride_st, stride_ib, stride_it,
                                 num_batchs, num_tokens, BLOCK: tl.constexpr,
                                 BLOCK_N: tl.constexpr):
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in range(0, num_tokens, BLOCK_N):
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        # Totalize the label address independently of floating-point results.
        indices = tl.load(Indices + off * stride_ib + tl.minimum(valid_pos, num_tokens - 1) * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)


@triton.jit
def _multinomial_sampling_unrolled_kernel(Scores, Seeds, Offsets, Indices, Outputs,
                                 stride_sb, stride_st, stride_ib, stride_it,
                                 num_batchs, num_tokens: tl.constexpr, BLOCK: tl.constexpr,
                                 BLOCK_N: tl.constexpr):
    """Kernel."""
    batch_block_id = tl.program_id(0)

    off = batch_block_id * BLOCK + tl.arange(0, BLOCK)
    n_off = tl.arange(0, BLOCK_N)

    off_mask = off < num_batchs
    seed = tl.load(Seeds + off, mask=off_mask)
    offset = tl.load(Offsets + off, mask=off_mask).to(tl.int32)

    samp = tl.rand(seed, offset)[:, None]
    acc = tl.zeros((BLOCK, ), dtype=tl.float32)
    output = tl.load(Indices + off * stride_ib, mask=off_mask)

    for b_idx in tl.static_range(tl.cdiv(num_tokens, BLOCK_N)):
        b_idx = b_idx * BLOCK_N
        s_off = b_idx + n_off
        s_mask = off_mask[:, None] & (s_off[None, :] < num_tokens)
        scores = tl.load(Scores + off[:, None] * stride_sb +
                         s_off[None, :] * stride_st,
                         mask=s_mask,
                         other=0.0).to(tl.float32)
        c_scores = tl.cumsum(scores, 1)
        cum_scores = acc[:, None] + c_scores
        acc += tl.max(c_scores, 1)

        pre_cum_scores = cum_scores - scores
        valid_mask = (samp > pre_cum_scores) & (samp <= cum_scores)
        found_mask = tl.sum(valid_mask, 1) > 0

        valid_pos = b_idx + tl.argmax(valid_mask.to(tl.int32), 1)
        # Totalize the label address independently of floating-point results.
        indices = tl.load(Indices + off * stride_ib + tl.minimum(valid_pos, num_tokens - 1) * stride_it,
                          mask=found_mask & off_mask,
                          other=-1)
        output = tl.where(found_mask, indices, output)

    tl.store(Outputs + off, output, mask=off_mask)


def _tensor(tensor, name, dtype, device=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tensor.layout != torch.strided or tensor.dtype != dtype:
        raise TypeError(f"{name} has unsupported layout or dtype")
    if tensor.device.type != "cuda" or (device is not None and tensor.device != device):
        raise ValueError(f"{name} must be on the scores CUDA device")
    if not tensor.is_contiguous() or tensor.is_neg() or tensor.is_conj():
        raise ValueError(f"{name} must be contiguous with ordinary view metadata")


def _address_window(tensor, last, width):
    base = tensor.data_ptr()
    if base % width or base < 0 or base + width * last >= 1 << 64:
        raise ValueError("formed pointer exceeds the aligned uint64 address domain")


def _probability_rows(scores):
    # Every finite FP32 value is an integer multiple of 2^-149. Python's
    # float conversion is exact here; all summation below uses integers.
    unit = 1 << 149
    for row in scores.detach().cpu().tolist():
        total = 0
        for value in row:
            if not math.isfinite(value) or value < 0:
                raise ValueError("scores must be finite and nonnegative")
            numerator, denominator = value.as_integer_ratio()
            total += numerator * (unit // denominator)
        if total != unit:
            raise ValueError("each FP32 row must sum exactly to one")


def multinomial_sampling(scores: torch.Tensor,
                         seeds: torch.LongTensor,
                         offsets: torch.LongTensor,
                         indices: torch.Tensor = None):
    """Sample exactly normalized FP32 rows with the source Philox10 graph."""
    _tensor(scores, "scores", torch.float32)
    if scores.dim() != 2:
        raise ValueError("scores must have rank two")
    batch_size, num_tokens = scores.size()
    device = scores.device
    if not (0 <= batch_size < 1 << 31 and 0 < num_tokens < 1 << 31):
        raise ValueError("shape exceeds the nonempty signed32 source domain")

    if indices is not None:
        _tensor(indices, "indices", torch.int64, device)
        if indices.dim() != 2 or indices.size() != scores.size():
            raise ValueError("indices must have the scores shape")
    _probability_rows(scores)

    if indices is None:
        indices = torch.arange(num_tokens, device=device)
        indices = indices.expand_as(scores)

    if num_tokens == 1:
        return indices.clone()

    _tensor(seeds, "seeds", torch.int64, device)
    _tensor(offsets, "offsets", torch.int64, device)
    if seeds.shape != (batch_size,) or offsets.shape != (batch_size,):
        raise ValueError("seeds and offsets must have one entry per row")
    outputs = indices[:, 0].clone()

    BLOCK = 8
    BLOCK_N = 128

    # Include the inactive lanes: the unchanged source still forms their
    # pointers before applying load/store masks.
    rows = triton.cdiv(batch_size, BLOCK) * BLOCK
    tokens = triton.cdiv(num_tokens, BLOCK_N) * BLOCK_N
    if rows >= 1 << 31 or tokens >= 1 << 31:
        raise ValueError("padded launch exceeds signed32 source indices")
    last_row = max(rows - 1, 0)
    last_token = tokens - 1
    for tensor, width in ((scores, 4), (indices, 8)):
        sb, st = tensor.stride()
        last = last_row * sb + last_token * st
        if not (0 <= sb < 1 << 31 and 0 <= st < 1 << 31 and last < 1 << 31):
            raise ValueError("strided padded index exceeds signed32")
        _address_window(tensor, last, width)
    for tensor in (seeds, offsets, outputs):
        _address_window(tensor, last_row, 8)

    grid = [triton.cdiv(batch_size, BLOCK)]
    kernel_meta = get_kernel_meta(scores)
    with torch.cuda.device(scores.device):
        _multinomial_sampling_kernel[grid](scores,
                                           seeds,
                                           offsets,
                                           indices,
                                           outputs,
                                           stride_sb=scores.stride(0),
                                           stride_st=scores.stride(1),
                                           stride_ib=indices.stride(0),
                                           stride_it=indices.stride(1),
                                           num_batchs=batch_size,
                                           num_tokens=num_tokens,
                                           BLOCK=BLOCK,
                                           BLOCK_N=BLOCK_N,
                                           num_warps=8,
                                           **kernel_meta)

    return outputs



"""Separate FP32 repair candidate; the original benchmark is unchanged."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _per_block_int8_repaired(
    X, X_int8, Scale, L: tl.constexpr, C: tl.constexpr,
    BLK: tl.constexpr, BLOCKS: tl.constexpr, BASE: tl.constexpr,
    WIDTH: tl.constexpr, IS_Q: tl.constexpr,
):
    block = tl.program_id(0).to(tl.int64) + BASE
    batch = block // BLOCKS
    local_block = block % BLOCKS
    lane = tl.arange(0, WIDTH).to(tl.int64)
    row = local_block * BLK + lane // C
    offset = batch * L * C + local_block * BLK * C + lane
    active = (lane < BLK * C) & (row < L)
    x = tl.load(X + offset, mask=active, other=0.0)
    if IS_Q:
        x = x * (C**-0.5 * 1.44269504)
    finite = tl.abs(x) <= 3.4028234663852886e38
    bad = tl.max(tl.where(finite, 0, 1), 0)
    x = tl.where(finite, x, 0.0)
    maximum = tl.max(tl.abs(x), 0)
    scale = maximum / 127.0
    usable = (bad == 0) & (scale > 0.0)
    denominator = tl.where(usable, scale, 1.0)
    normalized = x / denominator
    normalized = tl.where(usable, normalized, 0.0)
    normalized = tl.minimum(tl.maximum(normalized, -127.0), 127.0)
    biased = normalized + 0.5 * tl.where(normalized >= 0.0, 1.0, -1.0)
    quantized = biased.to(tl.int8)
    tl.store(X_int8 + offset, quantized, mask=active)
    tl.store(Scale + block, tl.where(bad == 0, scale, 0.0))


def _quantize_one(x, block, is_q):
    if not isinstance(x, torch.Tensor):
        raise TypeError("q and k must be torch tensors")
    if x.dim() not in (3, 4):
        raise ValueError("q and k must independently have rank 3 or 4")
    if x.layout != torch.strided or x.dtype != torch.float32:
        raise TypeError("this candidate implements the benchmark FP32 input route")
    if not x.is_cuda:
        raise ValueError("q and k must be CUDA tensors")
    if not x.is_contiguous() or x.is_neg():
        raise ValueError("inputs must be contiguous with no unresolved negative bit")
    if not isinstance(block, int) or isinstance(block, bool) or block <= 0:
        raise ValueError("block sizes must be positive integers")
    length, channels = x.shape[-2:]
    blocks = triton.cdiv(length, block)
    batches = math.prod(x.shape[:-2])
    if block * channels > 2**20:
        raise ValueError("block times channels exceeds the Triton tile limit")
    if x.numel() >= 2**61 or batches * blocks >= 2**61:
        raise ValueError("input and scale byte extents must fit signed int64")
    if x.numel() > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("input data pointer must be FP32 aligned")
    with torch.cuda.device(x.device):
        output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        scales = torch.empty((*x.shape[:-2], blocks, 1), device=x.device,
                             dtype=torch.float32)
        if batches == 0 or length == 0:
            return output, scales
        if channels == 0:
            scales.zero_()
            return output, scales
        prepared = x
        width = triton.next_power_of_2(block * channels)
        # Chunk only the launch grid; all pointer arithmetic is widened first.
        programs = batches * blocks
        for base in range(0, programs, 2147483647):
            count = min(programs - base, 2147483647)
            _per_block_int8_repaired[(count,)](
                prepared, output, scales, L=length, C=channels, BLK=block,
                BLOCKS=blocks, BASE=base, WIDTH=width, IS_Q=is_q,
                num_warps=4, enable_fp_fusion=False)
        return output, scales


def per_block_int8_repaired(q, k, BLKQ=128, BLKK=64):
    q_int8, q_scale = _quantize_one(q, BLKQ, True)
    k_int8, k_scale = _quantize_one(k, BLKK, False)
    return q_int8, q_scale, k_int8, k_scale

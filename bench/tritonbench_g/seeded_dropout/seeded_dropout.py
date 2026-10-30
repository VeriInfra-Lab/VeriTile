"""Checked FP32 contiguous seeded dropout; kernel and original host body retained.

No input copy, RNG substitute, finite-payload restriction, or denominator guard.
The source still eagerly divides when a Python p below one rounds to FP32 one.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout_index_commuted(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.layout != torch.strided:
        raise TypeError("x must have strided layout")
    if x.dtype != torch.float32:
        raise TypeError("x must have dtype torch.float32")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.is_neg():
        raise ValueError("x must not have an unresolved negative view bit")
    if x.numel() >= 1 << 31:
        raise ValueError("numel must be smaller than 2**31")
    if x.numel() > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("nonempty x must have a 4-byte-aligned data pointer")
    if type(p) is not float:
        raise TypeError("p must be a Python float")
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be finite and satisfy 0 <= p < 1")
    if type(seed) is not int:
        raise TypeError("seed must be a Python int")
    if not -(1 << 63) <= seed < 1 << 64:
        raise ValueError("seed must be in [-2**63, 2**64)")

    with torch.cuda.device(x.device):
        output = torch.empty_like(x)
        assert x.is_contiguous()
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
        return output

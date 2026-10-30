"""Checked source-precision masked selection; original kernel and autotune retained."""
import logging
import math

import torch
import triton
import triton.language as tl
from typing import Tuple


Shape = Tuple[int]

def broadcastable(s1: Shape, s2: Shape) -> bool:
    r1 = len(s1)
    if r1 == 0:
        return True
    r2 = len(s2)
    if r2 == 0:
        return True

    s1, s2 = (s1, s2) if r1 >= r2 else (s2, s1)
    r1, r2 = (r1, r2) if r1 >= r2 else (r2, r1)

    d = r1 - r2
    for i in range(r2):
        if s1[d + i] == 1 or s2[i] == 1 or s1[d + i] == s2[i]:
            continue
        return False
    return True


def cfggen():
    configs = [
        triton.Config({"BLOCK_SIZE": bs}, num_warps=w)
        for w in [4, 8, 16, 32]
        for bs in [256, 512, 1024, 2048, 4096]
    ]
    return configs



@triton.autotune(configs=cfggen(), key=["n_elements"])
@triton.jit
def masked_select_kernel(
    inp_ptr,
    select_mask_ptr,
    prefix_sum_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    inp = tl.load(inp_ptr + offsets, mask=mask, other=0.0)
    select_mask = tl.load(select_mask_ptr + offsets, mask=mask, other=0.0).to(tl.int1)
    out_offset = tl.load(prefix_sum_ptr + offsets, mask=mask, other=0.0) - 1

    tl.store(out_ptr + out_offset, inp, mask=(select_mask and mask))


def masked_select(inp, mask):
    if not isinstance(inp, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError("input and mask must be tensors")
    if inp.layout != torch.strided or mask.layout != torch.strided:
        raise TypeError("input and mask must be strided")
    if inp.dtype != torch.float32 or mask.dtype != torch.bool:
        raise TypeError("input must be FP32 and mask must be Bool")
    if not inp.is_cuda or not mask.is_cuda:
        raise ValueError("input and mask must be CUDA tensors")
    if inp.device != mask.device:
        raise ValueError("input and mask must use the same device")
    if mask.is_neg():
        raise ValueError("Bool mask must not have a negative view bit")
    if not broadcastable(tuple(inp.shape), tuple(mask.shape)):
        raise ValueError("input and mask shapes must broadcast")
    shape = torch.broadcast_shapes(inp.shape, mask.shape)
    n = math.prod(shape)
    if n == 0 or n >= (1 << 31):
        raise ValueError("broadcasted numel must satisfy 0 < numel < 2**31")

    logging.debug("GEMS MASKED SELECT")

    inp_shape = tuple(inp.shape)
    mask_shape = tuple(mask.shape)

    assert broadcastable(
        inp_shape, mask_shape
    ), "The shapes of the `mask` and the `input` tensor must be broadcastable"
    inp, mask = torch.broadcast_tensors(inp, mask)

    inp = inp.resolve_neg().contiguous()
    mask = mask.contiguous()

    mask_flattened = mask.ravel()

    prefix_sum = mask_flattened.cumsum(axis=0)
    if inp.data_ptr() % 4 != 0:
        raise ValueError("prepared FP32 input must have a 4-byte-aligned data pointer")
    out = torch.empty(prefix_sum[-1].item(), dtype=inp.dtype, device=inp.device)

    n_elements = inp.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(inp.device):
        masked_select_kernel[grid](inp, mask_flattened, prefix_sum, out, n_elements)
    return out

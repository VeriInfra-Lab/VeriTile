import logging
import math
import operator
from collections import namedtuple

import torch
import triton
import triton.language as tl


@triton.jit
def max_kernel_1(
    inp: tl.tensor,
    mid: tl.tensor,
    M: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)


@triton.jit
def max_kernel_2(mid: tl.tensor, out: tl.tensor, mid_size: tl.int32, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID).to(tl.int64)
    mid_ptrs = mid + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 8}, num_warps=8),
        triton.Config({"BLOCK_M": 16}, num_warps=8),
        triton.Config({"BLOCK_M": 32}, num_warps=8),
    ],
    key=[
        "M",
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_N": heur_block_n,
    }
)
@triton.jit
def max_kernel(
    inp: tl.tensor,
    out_value: tl.tensor,
    out_index: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    offset = m_offset[:, None] * tl.cast(N, tl.int64) * tl.cast(K, tl.int64) + n_offset[None, :] * tl.cast(K, tl.int64) + pid_k
    offset_index = m_offset * tl.cast(K, tl.int64) + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    result_value, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


def _check_input(inp, require_contiguous):
    if not isinstance(inp, torch.Tensor):
        raise TypeError("input must be a tensor")
    if inp.layout != torch.strided or inp.dtype != torch.float32 or not inp.is_cuda:
        raise ValueError("input must be a strided FP32 CUDA tensor")
    if inp.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if require_contiguous and not inp.is_contiguous():
        raise ValueError("global max requires a contiguous input")
    if inp.numel() >= 2**31:
        raise ValueError("input must contain fewer than 2**31 elements")
    if inp.numel() and inp.data_ptr() % 4:
        raise ValueError("nonempty FP32 input must be 4-byte aligned")


def max(inp):
    _check_input(inp, True)
    M = inp.numel()
    if M == 0:
        raise ValueError("global maximum requires a nonempty input")
    block_size = triton.next_power_of_2(math.isqrt(M - 1) + 1)
    mid_size = triton.cdiv(M, block_size)
    block_mid = triton.next_power_of_2(mid_size)
    with torch.cuda.device(inp.device):
        mid = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
        out = torch.empty([], dtype=inp.dtype, device=inp.device)
        max_kernel_1[(mid_size, 1, 1)](inp, mid, M, block_size)
        max_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
    return out


def max_dim(inp, dim=None, keepdim=False):
    _check_input(inp, False)
    dim = operator.index(dim)
    if not -inp.ndim <= dim < inp.ndim:
        raise ValueError("invalid dimension")
    dim %= inp.ndim
    shape = inp.shape
    N = shape[dim]
    if not 0 < N <= 32768:
        raise ValueError("reduction axis must have between 1 and 32768 elements")
    M = math.prod(shape[:dim])
    K = math.prod(shape[dim+1:])
    if M and K > 65535:
        raise ValueError("nonempty trailing launch dimension must fit CUDA grid.y")
    # This contiguous() operation belongs to the original max_dim wrapper.
    inp = inp.contiguous()
    shape_list = list(shape)
    shape_list[dim] = 1
    with torch.cuda.device(inp.device):
        out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
        out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
        if not keepdim:
            out_value = torch.squeeze(out_value, dim)
            out_index = torch.squeeze(out_index, dim)
        if M and K:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)
            max_kernel[grid](inp, out_value, out_index, M, N, K)
    return namedtuple("max", ["values", "indices"])(out_value, out_index)

"""Separate log-softmax repair; selected proof scope is FP32 in/out.

The checked wrapper admits only FP32 storage. Both kernels retain every autotune and
warp-count branch; no runtime grid-compatibility expansion is claimed.
"""
import logging
import math
import operator

import torch
import triton
import triton.language as tl

ORIGINAL_SHA256 = "ea7ae1e49d9dac09bb0fd692bebc73af80710370a05365f204f0a4213809ddaf"


def heur_block_n(args):
    return triton.next_power_of_2(args["N"])


def heur_num_warps(args):
    if args["N"] <= 1024:
        return 4
    elif args["N"] <= 2048:
        return 8
    else:
        return 16


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 1}), triton.Config({"BLOCK_M": 2}),
             triton.Config({"BLOCK_M": 4}), triton.Config({"BLOCK_M": 8})],
    key=["M", "N"],
)
@triton.heuristics({"BLOCK_N": heur_block_n, "num_warps": heur_num_warps})
@triton.jit
def log_softmax_kernel_repaired(
    output_ptr, input_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    m64, n64, k64 = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    # Even masked lanes form an in-bounds integer offset. A successful dense
    # FP32 allocation bounds the entire product, including these intermediates.
    safe_m = tl.minimum(m_offset, m64 - 1)
    safe_n = tl.minimum(n_offset, n64 - 1)
    safe_k = tl.minimum(pid_k, k64 - 1)
    offset = safe_m[:, None] * n64 * k64 + safe_n[None, :] * k64 + safe_k
    mask = (m_offset[:, None] < m64) & (n_offset[None, :] < n64) & (pid_k < k64)
    inp = tl.load(input_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    row_minus_max = inp - tl.max(inp, axis=1)[:, None]
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1)[:, None]
    # Compute log-softmax directly: log(exp(delta)) would underflow first.
    result = row_minus_max - tl.log(denominator)
    tl.store(output_ptr + offset, result, mask=mask)


@triton.autotune(
    configs=[triton.Config({"BLOCK_M": 1}), triton.Config({"BLOCK_M": 2}),
             triton.Config({"BLOCK_M": 4}), triton.Config({"BLOCK_M": 8})],
    key=["M", "N"],
)
@triton.heuristics({"BLOCK_N": heur_block_n, "num_warps": heur_num_warps})
@triton.jit
def log_softmax_backward_kernel_repaired(
    out_ptr, out_grad_ptr, in_grad_ptr, M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    m64, n64, k64 = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    safe_m = tl.minimum(m_offset, m64 - 1)
    safe_n = tl.minimum(n_offset, n64 - 1)
    safe_k = tl.minimum(pid_k, k64 - 1)
    offset = safe_m[:, None] * n64 * k64 + safe_n[None, :] * k64 + safe_k
    mask = (m_offset[:, None] < m64) & (n_offset[None, :] < n64) & (pid_k < k64)
    out = tl.load(out_ptr + offset, mask=mask, other=-float("inf")).to(tl.float32)
    out_grad = tl.load(out_grad_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(out_grad, axis=1)
    in_grad = out_grad - tl.exp(out.to(tl.float32)) * scale[:, None]
    tl.store(in_grad_ptr + offset, in_grad, mask=mask)


def _check_tensor(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("input must be a tensor")
    if x.layout != torch.strided or x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError("input must be a strided FP32 CUDA tensor")
    if x.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if x.numel() >= 2**31:
        raise ValueError("input must contain fewer than 2**31 elements")
    if x.numel() and x.data_ptr() % 4:
        raise ValueError("nonempty FP32 input must be 4-byte aligned")


def _dimensions(x, dim):
    dim = operator.index(dim)
    if dim < -x.ndim or dim >= x.ndim:
        raise ValueError("invalid dimension")
    dim %= x.ndim
    M = math.prod(x.shape[:dim])
    N = x.shape[dim]
    K = math.prod(x.shape[dim+1:])
    if x.numel() and (N > 131072 or K > 65535):
        raise ValueError("nonempty shape exceeds retained tile or CUDA grid.y bounds")
    return dim, M, N, K


class LogSoftmaxRepaired(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, dtype):
        _check_tensor(x)
        if dtype not in (None, torch.float32):
            raise ValueError("output dtype must be FP32")
        dim, M, N, K = _dimensions(x, dim)
        inp = x.contiguous()
        out = torch.empty_like(inp, dtype=torch.float32)
        if inp.numel():
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)
            with torch.cuda.device(inp.device):
                log_softmax_kernel_repaired[grid](out, inp, M, N, K, enable_fp_fusion=True)
        ctx.save_for_backward(out)
        ctx.dim = dim
        return out

    @staticmethod
    def backward(ctx, out_grad):
        (out,) = ctx.saved_tensors
        _check_tensor(out)
        _check_tensor(out_grad)
        if out_grad.shape != out.shape or out_grad.device != out.device:
            raise ValueError("upstream gradient must match saved output shape and device")
        if not out.is_contiguous():
            raise ValueError("saved output must remain contiguous")
        dim, M, N, K = _dimensions(out, ctx.dim)
        out_grad = out_grad.contiguous()
        in_grad = torch.empty_like(out)
        if out.numel():
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), K)
            with torch.cuda.device(in_grad.device):
                log_softmax_backward_kernel_repaired[grid](
                    out, out_grad, in_grad, M, N, K, enable_fp_fusion=True)
        return in_grad, None, None


def log_softmax_repaired(x, dim=-1, dtype=None):
    return LogSoftmaxRepaired.apply(x, dim, dtype)


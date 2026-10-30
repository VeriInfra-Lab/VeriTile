"""The original FP16 transpose with an explicit single-tile shape contract."""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(
    M: tl.tensor,
    Out: tl.tensor,
    matrix_stridex: tl.int32,
    matrix_stridey: tl.int32,
    out_stridex: tl.int32,
    out_stridey: tl.int32,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    matrix_ptr = M + size_m_arange[:, None] * matrix_stridex + d_head_arange[None, :] * matrix_stridey
    out_ptr = Out + d_head_arange[:, None] * out_stridex + size_m_arange[None, :] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, tl.trans(matrix))


@triton.jit
def kernel_index_commuted(
    M: tl.tensor,
    Out: tl.tensor,
    matrix_stridex: tl.int32,
    matrix_stridey: tl.int32,
    out_stridex: tl.int32,
    out_stridey: tl.int32,
    SIZE_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    """The existing Lean variant, commuting only integer index multiplication."""
    size_m_arange = tl.arange(0, SIZE_M)
    d_head_arange = tl.arange(0, D_HEAD)
    matrix_ptr = M + matrix_stridex * size_m_arange[:, None] + d_head_arange[None, :] * matrix_stridey
    out_ptr = Out + d_head_arange[:, None] * out_stridex + size_m_arange[None, :] * out_stridey
    matrix = tl.load(matrix_ptr)
    tl.store(out_ptr, tl.trans(matrix))


def wrapper(size_m, d_head):
    if type(size_m) is not int or type(d_head) is not int:
        raise TypeError("dimensions must be built-in integers")
    if size_m <= 0 or d_head <= 0:
        raise ValueError("dimensions must be positive")
    if size_m & (size_m - 1) or d_head & (d_head - 1):
        raise ValueError("dimensions must be powers of two")
    if size_m * d_head > 2**20:
        raise ValueError("the single tile may contain at most 2**20 elements")

    matrix = torch.randn((size_m, d_head), dtype=torch.float16, device="cuda")
    out = torch.zeros((d_head, size_m), dtype=torch.float16, device="cuda")
    grid = (1,)
    kernel[grid](matrix, out, *matrix.stride(), *out.stride(), size_m, d_head)
    return out

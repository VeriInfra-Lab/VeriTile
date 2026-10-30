"""Guarded source-selected FP32 SwiGLU; original benchmark is preserved."""

import torch
import triton
import triton.language as tl

# Forward Triton kernel for Swish-Gated Linear Units (Swiglu)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}),
        triton.Config({'BLOCK_N': 64}),
        triton.Config({'BLOCK_N': 128}),
        triton.Config({'BLOCK_N': 256}),
        triton.Config({'BLOCK_N': 512}),
        triton.Config({'BLOCK_N': 1024}),
    ],
    key=['ncols'],
)
@triton.jit
def _swiglu_fwd_kernel(
    X, Y, OUT, stride_x_row, stride_y_row, stride_out_row, ncols, BLOCK_N: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    OUT += row * stride_out_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)

# Function to invoke the forward kernel
def _check_payload(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError("expected strided FP32 storage")
    if not tensor.is_cuda:
        raise ValueError("expected CUDA storage")
    if tensor.is_neg():
        raise ValueError("negative views are unsupported")


def swiglu_fwd_checked(xy, out=None):
    _check_payload(xy)
    if out is not None:
        _check_payload(out)
    if xy.ndim == 0 or xy.shape[-1] == 0 or xy.shape[-1] % 2 != 0:
        raise ValueError("expected a positive even final dimension")
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.reshape(-1, out.shape[-1])
        assert out.shape == x.shape
    assert out.stride(-1) == 1
    M, N = x.shape
    sx, so = x.stride(0), out.stride(0)
    if out.device != x.device:
        raise ValueError("input and output must be on the same device")
    if not (M < 2**31 and sx < 2**31 and so < 2**31 and
            M * sx + N + 1024 < 2**31 and M * so + N + 1024 < 2**31):
        raise ValueError("signed int32 launch metadata bound exceeded")
    if M > 1 and so < N:
        raise ValueError("output rows overlap")
    if M > 0:
        if x.data_ptr() % 4 != 0 or out.data_ptr() % 4 != 0:
            raise ValueError("active FP32 pointers must be 4-byte aligned")
        x_begin, out_begin = x.data_ptr(), out.data_ptr()
        x_end = x_begin + 4 * ((M - 1) * sx + 2 * N)
        out_end = out_begin + 4 * ((M - 1) * so + N)
        if not (out_end <= x_begin or x_end <= out_begin):
            raise ValueError("output span overlaps the live input span")
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_N']))
    with torch.cuda.device(x.device.index):
        _swiglu_fwd_kernel[grid](x, y, out, x.stride(0), y.stride(0), out.stride(0), N)
    return out.reshape(*batch_shape, out.shape[-1])



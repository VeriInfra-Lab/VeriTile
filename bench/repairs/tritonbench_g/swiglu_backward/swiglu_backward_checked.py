"""Checked FP32 source entry; original benchmark remains unchanged."""

import torch
import triton
import triton.language as tl

# Backward Triton kernel for Swish-Gated Linear Units (Swiglu)
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
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["OUT"] is not None})
@triton.jit
def _swiglu_bwd_kernel(
    X, Y, DOUT, OUT, DX, DY, stride_x_row, stride_y_row, stride_dout_row,
    stride_out_row, stride_dx_row, stride_dy_row, ncols, BLOCK_N: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += row * stride_x_row
    Y += row * stride_y_row
    DOUT += row * stride_dout_row
    if RECOMPUTE_OUTPUT:
        OUT += row * stride_out_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    dout = tl.load(DOUT + cols, mask=cols < ncols, other=0.).to(tl.float32)
    x_sigmoid = tl.sigmoid(x)
    dx = x_sigmoid * (1 + x * (1 - x_sigmoid)) * y * dout
    dy = x * x_sigmoid * dout
    tl.store(DX + cols, dx, mask=cols < ncols)
    tl.store(DY + cols, dy, mask=cols < ncols)
    if RECOMPUTE_OUTPUT:
        out = x * x_sigmoid * y
        tl.store(OUT + cols, out, mask=cols < ncols)

def _check_payload(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError("expected strided FP32 storage")
    if not tensor.is_cuda:
        raise ValueError("expected CUDA storage")
    if tensor.is_neg():
        raise ValueError("negative views are unsupported")


# Function to invoke the backward kernel
def swiglu_backward_checked(xy, dout, dxy=None, recompute_output=False, out=None):
    _check_payload(xy)
    _check_payload(dout)
    if dxy is not None:
        _check_payload(dxy)
    if recompute_output and out is not None:
        _check_payload(out)
    if xy.ndim == 0 or xy.shape[-1] == 0 or xy.shape[-1] % 2 != 0:
        raise ValueError("expected a positive even final dimension")
    if xy.stride(-1) != 1:
        xy = xy.contiguous()
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch_shape = xy.shape[:-1]
    xy = xy.reshape(-1, xy.shape[-1])
    x, y = xy.chunk(2, dim=-1)
    dout = dout.reshape(-1, dout.shape[-1])
    assert dout.shape == x.shape
    if dxy is None:
        dxy = torch.empty_like(xy)
    else:
        dxy = dxy.reshape(-1, dxy.shape[-1])
        assert dxy.shape == xy.shape
    dx, dy = dxy.chunk(2, dim=-1)
    assert dx.stride(-1) == 1
    assert dy.stride(-1) == 1
    if recompute_output:
        if out is None:
            out = torch.empty_like(x)
        else:
            out = out.reshape(-1, out.shape[-1])
            assert out.shape == x.shape
        assert out.stride(-1) == 1
    M, N = x.shape
    active = [x, y, dout, dx, dy] + ([out] if recompute_output else [])
    if any(t.device != x.device for t in active):
        raise ValueError("active tensors must share one device")
    if not (M < 2**31 and N < 2**31 and all(
            t.stride(0) < 2**31 and M * t.stride(0) + N + 1024 < 2**31 for t in active)):
        raise ValueError("signed int32 launch metadata bound exceeded")
    if M > 1 and dxy.stride(0) < 2 * N:
        raise ValueError("gradient rows overlap")
    if recompute_output and M > 1 and out.stride(0) < N:
        raise ValueError("output rows overlap")
    if M > 0:
        if any(t.data_ptr() % 4 != 0 for t in active):
            raise ValueError("active FP32 pointers must be 4-byte aligned")
        def live_span(t):
            start = t.data_ptr()
            return start, start + 4 * ((M - 1) * t.stride(0) + t.shape[1])
        def overlaps(a, b):
            a_start, a_end = live_span(a)
            b_start, b_end = live_span(b)
            return a_start < b_end and b_start < a_end
        if any(overlaps(dxy, t) for t in (xy, dout)):
            raise ValueError("gradient byte range must not overlap live inputs")
        if recompute_output and any(overlaps(out, t) for t in (xy, dout, dxy)):
            raise ValueError("output byte range must not overlap inputs or gradients")
    grid = lambda META: (M, triton.cdiv(N, META['BLOCK_N']))
    with torch.cuda.device(x.device.index):
        _swiglu_bwd_kernel[grid](
            x, y, dout, out if recompute_output else None, dx, dy, x.stride(0), y.stride(0),
            dout.stride(0), out.stride(0) if recompute_output else 0, dx.stride(0),
            dy.stride(0), N
        )
    if not recompute_output:
        return dxy.reshape(*batch_shape, dxy.shape[-1])
    else:
        return dxy.reshape(*batch_shape, dxy.shape[-1]), out.reshape(*batch_shape, out.shape[-1])




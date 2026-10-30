"""Row-index commutation candidate for the repaired FP32 backward kernel.

Only the three row/stride integer products commute. The production wrapper
continues to dispatch the original repaired kernel.
"""
import torch
import triton
import triton.language as tl

# Keep the native lane layout independent of view alignment and scalar values.
@triton.jit(
    do_not_specialize=["X", "DY", "DX", "stride_x_row", "stride_x_col",
                       "stride_dy_row", "stride_dy_col", "stride_dx_row",
                       "stride_dx_col", "N", "eps"],
    do_not_specialize_on_alignment=["X", "DY", "DX", "stride_x_row", "stride_x_col",
                                    "stride_dy_row", "stride_dy_col", "stride_dx_row",
                                    "stride_dx_col", "N", "eps"],
)
def _l2_norm_bwd_kernel_index_commuted(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row: tl.int64,
    stride_x_col: tl.int64,
    stride_dy_row: tl.int64,
    stride_dy_col: tl.int64,
    stride_dx_row: tl.int64,
    stride_dx_col: tl.int64,
    N: tl.int64,  # number of logical columns
    eps: tl.float32,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    X += stride_x_row * row
    DX += stride_dx_row * row
    DY += stride_dy_row * row
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols * stride_x_col, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols * stride_dy_col, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols * stride_dx_col, dx, mask=mask)


"""Existing Lean index-commutation candidate, transcribed to native Triton.

Original source remains the separately selected benchmark Python file.
"""
import triton
import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel_index_commuted(
    X, Y, OUT, stride_x_row, stride_y_row, stride_out_row, ncols, BLOCK_N: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += stride_x_row * row
    Y += stride_y_row * row
    OUT += stride_out_row * row
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(OUT + cols, out, mask=cols < ncols)

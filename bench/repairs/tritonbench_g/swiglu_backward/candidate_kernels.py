"""Native counterpart of the existing Lean commutedKernel candidate.

The selected historical source remains swiglu_backward_checked.py unchanged.
Only the six row-stride integer products are commuted.
"""
import triton
import triton.language as tl

@triton.jit
def swiglu_bwd_commuted(X, Y, DOUT, OUT, DX, DY, stride_x_row, stride_y_row, stride_dout_row, stride_out_row, stride_dx_row, stride_dy_row, ncols, BLOCK_N: tl.constexpr, RECOMPUTE_OUTPUT: tl.constexpr):
    row = tl.program_id(0)
    start_col = tl.program_id(1) * BLOCK_N
    X += stride_x_row * row
    Y += stride_y_row * row
    DOUT += stride_dout_row * row
    if RECOMPUTE_OUTPUT:
        OUT += stride_out_row * row
    DX += stride_dx_row * row
    DY += stride_dy_row * row
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

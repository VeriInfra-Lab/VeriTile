"""Existing Lean index-commutation candidate, transcribed to native Triton.

Original source remains the separately selected benchmark Python file.
"""
import triton
import triton.language as tl

@triton.jit
def _l2_norm_fwd_1pass_kernel_index_commuted(
    X, Y, N, eps, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    X += N * row
    Y += N * row
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)

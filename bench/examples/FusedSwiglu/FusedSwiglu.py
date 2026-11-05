"""New native source from the independent masked SwiGLU specification.
No recovered historical Python is claimed. Inputs are FP32, the intermediate
scratch is BF16, and both source roundings remain explicit. The candidate has
two ordered launches. Use enable_fp_fusion=False for separate arithmetic.
"""
import triton
import triton.language as tl

@triton.jit
def swiglu_fused(X, Y, OUT, ncols: tl.constexpr, BLOCK_N: tl.constexpr):
    start_col = tl.program_id(0) * BLOCK_N
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0)
    sil = (x * tl.sigmoid(x)).to(tl.bfloat16)
    out = sil * y
    tl.store(OUT + cols, out.to(tl.bfloat16), mask=cols < ncols)

@triton.jit
def silu_step(X, S, ncols: tl.constexpr, BLOCK_N: tl.constexpr):
    start_col = tl.program_id(0) * BLOCK_N
    cols = start_col + tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < ncols, other=0.0)
    s = x * tl.sigmoid(x)
    tl.store(S + cols, s.to(tl.bfloat16), mask=cols < ncols)

@triton.jit
def mul_step(S, Y, OUT, ncols: tl.constexpr, BLOCK_N: tl.constexpr):
    start_col = tl.program_id(0) * BLOCK_N
    cols = start_col + tl.arange(0, BLOCK_N)
    z = tl.load(S + cols, mask=cols < ncols, other=0.0).to(tl.float32)
    y = tl.load(Y + cols, mask=cols < ncols, other=0.0)
    out = z * y
    tl.store(OUT + cols, out.to(tl.bfloat16), mask=cols < ncols)

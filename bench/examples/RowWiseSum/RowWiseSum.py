"""New FP32 Python sources from the independent sum-of-a-row specification.

Each program reduces one complete tile of a row-major array. The comparison
reverses the input lanes; its FP equivalence retains the numerical rules and
intermediate-value guards in FPEquiv.lean. These are newly authored sources,
not recovered historical benchmark implementations.
"""
import triton
import triton.language as tl


@triton.jit
def row_wise_sum(xReg, yReg, nCol: tl.constexpr, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)


@triton.jit
def row_wise_sum_reversed(xReg, yReg, nCol: tl.constexpr, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = blockSize - 1 - tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)

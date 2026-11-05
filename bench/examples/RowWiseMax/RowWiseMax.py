"""New Python sources from the independent maximum-of-a-nonempty-row spec.

This example uses FP32 storage and constexpr row stride/tile width. Each
program writes the maximum of its complete input tile to one output cell.
The comparison inlines the load/reduction without changing either operation.
These functions are newly authored, not recovered historical source.
"""
import triton
import triton.language as tl


@triton.jit
def row_wise_max(xReg, yReg, nCol: tl.constexpr, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.max(values, axis=0)
    tl.store(yReg + row, result)


@triton.jit
def row_wise_max_inlined(xReg, yReg, nCol: tl.constexpr, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    tl.store(yReg + row, tl.max(tl.load(xReg + row * nCol + cols), axis=0))

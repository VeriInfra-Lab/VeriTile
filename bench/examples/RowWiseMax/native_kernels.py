"""New native source from the independent row-maximum specification (2026-10-24).
No historical Python source is claimed. Correct.lean specifies max of the
nonempty row window; the existing candidate inlines the reduction result into its store.
ABI: FP32 pointers, Int64 row stride, positive constexpr blockSize; launches
and backing memory must satisfy the existing row-window IO contract.
"""
import triton
import triton.language as tl


@triton.jit
def rowWiseMaxKernel(xReg, yReg, nCol: tl.int64, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.max(values, axis=0)
    tl.store(yReg + row, result)


@triton.jit
def inlinedKernel(xReg, yReg, nCol: tl.int64, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    tl.store(yReg + row, tl.max(values, axis=0))

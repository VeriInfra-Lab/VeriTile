"""New native source from the independent row-sum specification (2026-10-24).
No historical Python source is claimed. Correct.lean specifies sum of the
row window; the existing candidate reverses the input lane order. FP
reassociation remains subject to the existing finite-intermediate domain.
ABI: FP32 pointers, Int64 row stride, positive constexpr blockSize; launch
and backing memory satisfy the existing row-window IO contract. Lean also
retains its mathematical zero-length extension; that is not a GPU launch.
"""
import triton
import triton.language as tl


@triton.jit
def rowWiseSumKernel(xReg, yReg, nCol: tl.int64, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)


@triton.jit
def reversedKernel(xReg, yReg, nCol: tl.int64, blockSize: tl.constexpr):
    row = tl.program_id(0)
    cols = (blockSize - 1) - tl.arange(0, blockSize)
    values = tl.load(xReg + row * nCol + cols)
    result = tl.sum(values, axis=0)
    tl.store(yReg + row, result)

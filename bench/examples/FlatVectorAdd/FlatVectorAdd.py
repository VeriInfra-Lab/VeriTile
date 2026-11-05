"""New Python source derived from the independent masked-add specification.

No historical Python source is claimed. For each tile, write x[i] + y[i]
exactly where i < nElements; leave all other cells unchanged. Size arguments
are constexpr specializations in this worked example. The candidate only
commutes the sum, preserving the undefined masked-load values.
"""
import triton
import triton.language as tl


@triton.jit
def flat_vector_add(xReg, yReg, outReg, blockSize: tl.constexpr, nElements: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(xReg + offsets, mask=mask)
    y = tl.load(yReg + offsets, mask=mask)
    output = x + y
    tl.store(outReg + offsets, output, mask=mask)


@triton.jit
def flat_vector_add_commuted(xReg, yReg, outReg, blockSize: tl.constexpr, nElements: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(xReg + offsets, mask=mask)
    y = tl.load(yReg + offsets, mask=mask)
    output = y + x
    tl.store(outReg + offsets, output, mask=mask)

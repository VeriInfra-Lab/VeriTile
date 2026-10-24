"""Native Triton counterpart authored for this Lean-first worked example.

Use FP32 input and output pointers. Both sizes are constexpr, preserving this
example's existing specialization. This is not a historical benchmark source.
"""
import triton
import triton.language as tl


@triton.jit
def addKernelMasked(xReg, yReg, outReg, blockSize: tl.constexpr, nElements: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(xReg + offsets, mask=mask)
    y = tl.load(yReg + offsets, mask=mask)
    output = x + y
    tl.store(outReg + offsets, output, mask=mask)


@triton.jit
def optimizedKernel(x_ptr, y_ptr, out_ptr, nElements: tl.constexpr, blockSize: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * blockSize + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = y + x
    tl.store(out_ptr + offsets, output, mask=mask)

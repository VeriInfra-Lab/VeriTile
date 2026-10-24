"""Native Triton counterpart authored for this Lean-first worked example.

Use FP32 input and output pointers. blockSize is constexpr. This is a newly
published example source, not an independent historical benchmark original.
"""
import triton
import triton.language as tl


@triton.jit
def addKernel(xReg, yReg, outReg, blockSize: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(xReg + offs)
    y = tl.load(yReg + offs)
    out = x + y
    tl.store(outReg + offs, out)


@triton.jit
def optimizedKernel(x_ptr, y_ptr, out_ptr, blockSize: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * blockSize + tl.arange(0, blockSize)
    x = tl.load(x_ptr + offs)
    y = tl.load(y_ptr + offs)
    out = y + x
    tl.store(out_ptr + offs, out)

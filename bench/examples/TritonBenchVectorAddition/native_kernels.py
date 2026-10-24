"""Native Triton counterpart authored for the Lean-first worked example.

FP32 pointers; nElements and blockSize are constexpr in this example. The
selected vector_addition task has a separate runtime ABI and source review.
These functions are not that task's independent historical Python original.
"""
import triton
import triton.language as tl


@triton.jit
def originalKernel(x_ptr, y_ptr, output_ptr, nElements: tl.constexpr, blockSize: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * blockSize
    offsets = block_start + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def optimizedKernel(x_ptr, y_ptr, output_ptr, nElements: tl.constexpr, blockSize: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * blockSize
    offsets = block_start + tl.arange(0, blockSize)
    mask = offsets < nElements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)

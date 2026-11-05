"""New FP32 Python sources from the independent typed-add specification.

These newly authored functions are not historical benchmark recovery. Inputs
and output use FP32 storage; the explicit output conversion is part of both
bodies. The comparison changes only addition operand order.
"""
import triton
import triton.language as tl


@triton.jit
def float_add(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = x + y
    tl.store(out_ptr + offsets, output.to(tl.float32))


@triton.jit
def float_add_commuted(x_ptr, y_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    output = y + x
    tl.store(out_ptr + offsets, output.to(tl.float32))

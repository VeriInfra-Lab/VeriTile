"""Existing Lean index-commutation candidate, transcribed to native Triton.

Original source remains the separately selected benchmark Python file.
"""
import triton
import triton.language as tl

@triton.jit
def kldivergence_kernel_index_commuted(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)

"""Python counterpart of the pre-existing Lean reciprocal candidate.

Authored during source migration from that explicit transformation. This file
is not historical benchmark source; the original is softmax_triton1_repaired.py.
The only change is numerator / denominator -> numerator * (1 / denominator).
"""
import triton
import triton.language as tl

@triton.jit
def reciprocal_kernel(output_ptr: tl.tensor, input_ptr: tl.tensor, input_row_stride: tl.int64, input_col_stride: tl.int64, output_row_stride: tl.int64, output_col_stride: tl.int64, n_cols: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    input_ptrs = row_start_ptr + col_offsets * input_col_stride
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator * (1 / denominator)
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets * output_col_stride
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

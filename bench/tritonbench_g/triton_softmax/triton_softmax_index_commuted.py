"""Contiguous FP32 row softmax with complete row coverage."""
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride: tl.int64, input_col_stride: tl.int64, output_row_stride: tl.int64, output_col_stride: tl.int64, n_cols: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    row_start_ptr = input_ptr + input_row_stride * row_idx
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    columns = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    row = tl.load(row_start_ptr + columns * input_col_stride, mask=columns < n_cols, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(out_row_start_ptr + columns * output_col_stride, softmax_output, mask=columns < n_cols)

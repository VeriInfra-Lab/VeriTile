"""Contiguous FP32 row softmax with complete row coverage."""
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride: tl.int64, input_col_stride: tl.int64, output_row_stride: tl.int64, output_col_stride: tl.int64, n_cols: tl.int64, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(axis=0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride
    columns = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    row = tl.load(row_start_ptr + columns * input_col_stride, mask=columns < n_cols, other=-float('inf'))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    tl.store(out_row_start_ptr + columns * output_col_stride, softmax_output, mask=columns < n_cols)

def triton_softmax(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError('x must be a Tensor')
    if x.layout != torch.strided:
        raise ValueError('x must have strided layout')
    if x.dtype != torch.float32:
        raise ValueError('x must have FP32 dtype')
    if not x.is_cuda:
        raise ValueError('x must be on CUDA')
    if x.ndim != 2:
        raise ValueError('x must be rank two')
    if not x.is_contiguous():
        raise ValueError('x must be contiguous')
    if x.is_neg():
        raise ValueError('x must have no unresolved negative view bit')
    n_rows, n_cols = x.shape
    n = x.numel()
    if n and n_rows >= 2 ** 31:
        raise ValueError('nonempty row grid must be smaller than 2**31')
    if n and x.data_ptr() % 4:
        raise ValueError('x data pointer must be aligned to four bytes')
    if n == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 2 ** 20:
        raise ValueError("row block exceeds Triton's arange limit")
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    with torch.cuda.device(x.device):
        softmax_kernel[n_rows,](y, x, n_cols, 1, n_cols, 1, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y

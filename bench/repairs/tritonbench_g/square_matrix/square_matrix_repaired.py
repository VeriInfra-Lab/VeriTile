"""Elementwise square for contiguous rank-two FP32 CUDA tensors."""
import torch
import triton
import triton.language as tl

@triton.jit
def square_kernel(output_ptr: tl.tensor, input_ptr: tl.tensor, input_row_stride: tl.int32, output_row_stride: tl.int32, n_cols: tl.int32, BLOCK_SIZE: tl.constexpr):
    # Each program squares one matrix row.
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)


# Existing proof variant: commute only the integer row-stride multiplication.
@triton.jit
def square_index_commuted_kernel(output_ptr: tl.tensor, input_ptr: tl.tensor, input_row_stride: tl.int32, output_row_stride: tl.int32, n_cols: tl.int32, BLOCK_SIZE: tl.constexpr):
    # Each program squares one matrix row.
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + input_row_stride * row_idx
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)


def square(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.layout != torch.strided:
        raise ValueError("x must have strided layout")
    if x.dtype != torch.float32:
        raise ValueError("x must have FP32 dtype")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA")
    if x.ndim != 2:
        raise ValueError("x must be rank two")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.is_neg():
        raise ValueError("x must have no unresolved negative view bit")
    n_rows, n_cols = x.shape
    n = x.numel()
    if n >= 2**31:
        raise ValueError("x must contain fewer than 2**31 elements")
    if n and x.data_ptr() % 4:
        raise ValueError("x data pointer must be aligned to four bytes")
    if n == 0:
        return torch.empty_like(x, memory_format=torch.contiguous_format)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 2**20:
        raise ValueError("row block exceeds Triton's arange limit")
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    # Logical row width is valid for every contiguous matrix, including
    # singleton dimensions with noncanonical but irrelevant metadata strides.
    with torch.cuda.device(x.device):
        square_kernel[(n_rows,)](
            y, x, n_cols, n_cols, n_cols,
            num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE,
        )
    return y

"""Last-axis row softmax for contiguous FP32 CUDA matrices."""
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr: tl.tensor, input_ptr: tl.tensor, input_row_stride: tl.int64, input_col_stride: tl.int64, output_row_stride: tl.int64, output_col_stride: tl.int64, n_cols: tl.int64, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0).to(tl.int64)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    input_ptrs = row_start_ptr + col_offsets * input_col_stride
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=0)
    # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets * output_col_stride
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax(x):
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
    if n and n_rows >= 2**31:
        raise ValueError("nonempty row grid must be smaller than 2**31")
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
        softmax_kernel[(n_rows,)](
            y, x, n_cols, 1, n_cols, 1, n_cols,
            num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE,
        )
    return y

"""Last-axis FP32 softmax with optional additive FP32 mask."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride: tl.int64, n_cols: tl.int64, mask_ptr, BLOCK_SIZE: tl.constexpr):
    """the kernel function for implementing softmax operator
    Args:
        output_ptr: the output after finishing softmax operation, (N, hidden_dim)
        input_ptr: the tensor of input, shape should be (N, hidden_dim)
        n_cols(tl.constexpr): the number of cols of input
        BLOCK_SIZE(tl.constexpr): the block_size of your hidden_dim dimension, typically BLOCK_SIZE >= hidden_dim
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    if mask_ptr is not None:
        mask_ptrs = mask_ptr + row_idx * row_stride + col_offsets
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=0).to(tl.float32)
        row_minus_max = row_minus_max + mask
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def _check_tensor(x, name):
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if x.layout != torch.strided:
        raise ValueError(f"{name} must have strided layout")
    if x.dtype != torch.float32:
        raise ValueError(f"{name} must have FP32 dtype")
    if not x.is_cuda:
        raise ValueError(f"{name} must be on CUDA")
    if x.ndim == 0:
        raise ValueError(f"{name} must have a last axis")
    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if x.is_neg():
        raise ValueError(f"{name} must have no unresolved negative view bit")
    rows = 1
    for size in x.shape[:-1]:
        rows *= size
    cols, n = x.shape[-1], x.numel()
    if n and rows >= 2**31:
        raise ValueError("nonempty row grid must be smaller than 2**31")
    if n and x.data_ptr() % 4:
        raise ValueError(f"{name} data pointer must be aligned to four bytes")
    block = max(triton.next_power_of_2(cols), 2) if n else 0
    if n and block > 2**20:
        raise ValueError("row block exceeds Triton's arange limit")
    return rows, cols, n, block


def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    rows, cols, n, block = _check_tensor(input, "input")
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("softmax supports only the last axis")
    if mask is not None:
        mask_rows, mask_cols, _, _ = _check_tensor(mask, "mask")
        if mask.device != input.device:
            raise ValueError("mask and input must be on the same device")
        if mask_cols != cols:
            raise ValueError("mask and input must have the same last-axis width")
        if mask_rows != rows:
            raise ValueError("mask and input must have the same flattened row count")
    output = torch.empty_like(input, memory_format=torch.contiguous_format)
    if n == 0:
        return output
    num_warps = 16 if block >= 4096 else 8 if block >= 2048 else 4
    with torch.cuda.device(input.device):
        softmax_kernel[(rows,)](output, input, cols, cols, mask,
                               BLOCK_SIZE=block, num_warps=num_warps)
    return output


@triton.jit
def guarded_reciprocal_kernel(output_ptr, input_ptr, row_stride: tl.int64, n_cols: tl.int64, mask_ptr, BLOCK_SIZE: tl.constexpr):
    """the kernel function for implementing softmax operator
    Args:
        output_ptr: the output after finishing softmax operation, (N, hidden_dim)
        input_ptr: the tensor of input, shape should be (N, hidden_dim)
        n_cols(tl.constexpr): the number of cols of input
        BLOCK_SIZE(tl.constexpr): the block_size of your hidden_dim dimension, typically BLOCK_SIZE >= hidden_dim
    """
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf')).to(tl.float32)
    row_minus_max = row - tl.max(row, axis=0)
    if mask_ptr is not None:
        mask_ptrs = mask_ptr + row_idx * row_stride + col_offsets
        mask = tl.load(mask_ptrs, mask=col_offsets < n_cols, other=0).to(tl.float32)
        row_minus_max = row_minus_max + mask
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    finite_numerator = libdevice.finitef(numerator)
    finite_count = tl.sum(finite_numerator, axis=0)
    finite_denominator = libdevice.finitef(denominator)
    if (finite_count == BLOCK_SIZE) & finite_denominator & (denominator >= 1.0):
        softmax_output = numerator * (1.0 / denominator)
    else:
        softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)


def softmax_reciprocal(input: torch.Tensor, mask: torch.Tensor = None, dim=-1) -> torch.Tensor:
    rows, cols, n, block = _check_tensor(input, "input")
    if dim != -1 and dim != input.ndim - 1:
        raise ValueError("softmax supports only the last axis")
    if mask is not None:
        mask_rows, mask_cols, _, _ = _check_tensor(mask, "mask")
        if mask.device != input.device:
            raise ValueError("mask and input must be on the same device")
        if mask_cols != cols:
            raise ValueError("mask and input must have the same last-axis width")
        if mask_rows != rows:
            raise ValueError("mask and input must have the same flattened row count")
    output = torch.empty_like(input, memory_format=torch.contiguous_format)
    if n == 0:
        return output
    num_warps = 16 if block >= 4096 else 8 if block >= 2048 else 4
    with torch.cuda.device(input.device):
        guarded_reciprocal_kernel[(rows,)](output, input, cols, cols, mask,
                               BLOCK_SIZE=block, num_warps=num_warps)
    return output

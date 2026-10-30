
import torch
import triton
import triton.language as tl

@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


# Integer index-commutation candidate; floating arithmetic is unchanged.
@triton.jit
def _dropout_index_commuted(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout_repaired(x, x_keep, p):
    """Given-mask inverted dropout on contiguous FP32/int32 CUDA tensors."""
    if not isinstance(x, torch.Tensor) or not isinstance(x_keep, torch.Tensor):
        raise TypeError("x and x_keep must be tensors")
    if x.layout != torch.strided or x_keep.layout != torch.strided:
        raise ValueError("x and x_keep must use strided layouts")
    if x.dtype != torch.float32 or x_keep.dtype != torch.int32:
        raise ValueError("x must be FP32 and x_keep must be int32")
    if not (x.is_cuda and x_keep.is_cuda):
        raise ValueError("x and x_keep must be CUDA tensors")
    if x.device != x_keep.device:
        raise ValueError("x and x_keep must have the same device")
    if x.shape != x_keep.shape:
        raise ValueError("x and x_keep must have the same shape")
    if not (x.is_contiguous() and x_keep.is_contiguous()):
        raise ValueError("x and x_keep must be contiguous")
    if x.is_neg() or x_keep.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    n_elements = x.numel()
    if n_elements >= 2**31:
        raise ValueError("numel must be less than 2**31")
    if n_elements > 0 and (x.data_ptr() % 4 != 0 or x_keep.data_ptr() % 4 != 0):
        raise ValueError("nonempty inputs require 4-byte aligned data pointers")
    if type(p) is not float:
        raise TypeError("p must be a Python float")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    output = torch.empty_like(x)
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    with torch.cuda.device(x.device):
        _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output

"""FP32 sine for contiguous CUDA tensors under Contract.lean's checks."""
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_function(x_ptr: tl.tensor, output_ptr: tl.tensor, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    # x_ptr: pointer to input data
    # output_ptr: pointer to output data
    # n_elements: number of elements to process
    # BLOCK_SIZE: block size for Triton kernel
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def kernel_function_index_commuted(x_ptr: tl.tensor, output_ptr: tl.tensor, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    # x_ptr: pointer to input data
    # output_ptr: pointer to output data
    # n_elements: number of elements to process
    # BLOCK_SIZE: block size for Triton kernel
    pid = tl.program_id(0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.math.sin(x)
    tl.store(output_ptr + offsets, output, mask=mask)

def call_kernel(x):
    """Return a fresh FP32 elementwise sine; no input materialization."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.layout != torch.strided:
        raise TypeError("x must have strided layout")
    if x.dtype != torch.float32:
        raise TypeError("x must have dtype torch.float32")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.is_neg():
        raise ValueError("x must not have an unresolved negative view bit")
    n_elements = x.numel()
    if n_elements >= (1 << 31):
        raise ValueError("numel must be smaller than 2**31")
    if n_elements > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("nonempty x must have a 4-byte-aligned data pointer")
    with torch.cuda.device(x.device):
        output = torch.empty_like(x)
        if n_elements == 0:
            return output
        grid = ((n_elements + 1023) // 1024,)
        kernel_function[grid](x, output, n_elements, BLOCK_SIZE=1024)
        return output

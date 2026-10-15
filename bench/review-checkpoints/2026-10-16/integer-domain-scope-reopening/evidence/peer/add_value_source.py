import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def puzzle1_kernel(x_ptr: tl.tensor, output_ptr: tl.tensor, N: tl.int32, BLOCK_SIZE: tl.constexpr, value: tl.float32):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + value
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to call the kernel
def puzzle1(x: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and output.is_cuda
    N = output.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    return output





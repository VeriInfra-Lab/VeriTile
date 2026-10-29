import triton
import triton.language as tl
import torch


@triton.jit
def puzzle1_kernel(x_ptr: tl.tensor, output_ptr: tl.tensor, N: tl.int32,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x + 10
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def puzzle1_commuted_kernel(x_ptr: tl.tensor, output_ptr: tl.tensor, N: tl.int32,
                            BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    y = 10
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)


def puzzle1_repaired(x: torch.Tensor):
    """Add 10 to a contiguous FP32 CUDA tensor under the checked contract."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.layout != torch.strided:
        raise ValueError("x must use strided layout")
    if x.dtype != torch.float32:
        raise TypeError("x must have FP32 dtype")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.is_neg():
        raise ValueError("x must have no unresolved negative view bit")
    N = x.numel()
    if N >= 2**31:
        raise ValueError("x must contain fewer than 2**31 elements")
    if N > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("x must have a 4-byte aligned data pointer")
    output = torch.empty_like(x)
    if N == 0:
        return output
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    with torch.cuda.device(x.device):
        puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024)
    return output

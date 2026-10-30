"""ReLU with a checked contiguous FP32 CUDA contract and complete output coverage."""
import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor, N: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x >= 0, x, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def relu(x):
    """Return a fresh tensor. NaNs become +0; a selected -0 stays -0."""
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
    N = x.numel()
    if N >= (1 << 31):
        raise ValueError("numel must be smaller than 2**31")
    if N > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("nonempty x must have a 4-byte-aligned data pointer")
    with torch.cuda.device(x.device):
        out = torch.empty_like(x, dtype=torch.float32, device=x.device)
        if N == 0:
            return out
        BLOCK_SIZE = 1024
        relu_kernel[((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)](x, out, N, BLOCK_SIZE)
        return out


@triton.jit
def relu_inlined_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor, N: tl.constexpr, block_size: tl.constexpr):
    """Existing store-inline candidate; the wrapper selects relu_kernel."""
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.where(x >= 0, x, 0.0), mask=mask)

import triton
import triton.language as tl
import torch

@triton.jit
def relu_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor, N: tl.constexpr, block_size: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x >= 0, x, 0.0)
    if pid == 0:
        tl.store(out_ptr + offsets, result, mask=mask)

def relu(x):
    out = torch.empty_like(x, dtype=torch.float32, device=x.device)
    N = out.numel()
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](x, out, N, BLOCK_SIZE)
    return out


import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    in_ptr0: tl.tensor,
    in_ptr1: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def add_wrapper(x, y):
    out = torch.zeros_like(x)
    
    BLOCK_SIZE = 4
    n_elements = x.numel()

    # Calculate the number of blocks needed
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)

    return out

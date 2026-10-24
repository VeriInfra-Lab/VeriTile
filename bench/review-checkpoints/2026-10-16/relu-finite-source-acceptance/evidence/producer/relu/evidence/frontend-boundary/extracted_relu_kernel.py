import triton
import triton.language as tl
@triton.jit
def relu_kernel(x_ptr: tl.tensor, out_ptr: tl.tensor, N: tl.constexpr, block_size: tl.constexpr):
    # Get the index of the current thread
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < N

    # Load elements from global memory
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute linear layer
    result = tl.where(x >= 0, x, 0.0)

    # Write result to global memory
    if pid == 0:
        tl.store(out_ptr + offsets, result, mask=mask)

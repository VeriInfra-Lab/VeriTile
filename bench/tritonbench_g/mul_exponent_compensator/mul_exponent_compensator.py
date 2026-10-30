import torch
import triton
import triton.language as tl

# Triton kernel to multiply each element in the source tensor by a constant exponent compensator
@triton.jit
def mul_kernel_repaired(src, dst, n_elements: tl.int64, BLOCK_SIZE: tl.constexpr):
    # Define a constant exponent compensator
    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
    # Calculate the indices for the current program ID
    idxs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = idxs < n_elements
    # Load elements from the source tensor
    x = tl.load(src + idxs, mask=mask)
    # Multiply each element by the exponent compensator
    y = x * exponent_compensator
    # Store the result in the destination tensor
    tl.store(dst + idxs, y, mask=mask)


@triton.jit
def mul_kernel_scalar_left(src, dst, n_elements: tl.int64, BLOCK_SIZE: tl.constexpr):
    """Existing scalar-left variant with the same widened masked accesses."""
    exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)
    idxs = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = idxs < n_elements
    x = tl.load(src + idxs, mask=mask)
    y = exponent_compensator * x
    tl.store(dst + idxs, y, mask=mask)

def launch_mul_kernel(src, BLOCK_SIZE=1):
    if not isinstance(src, torch.Tensor):
        raise TypeError("src must be a Tensor")
    # Preserve the original current-CUDA allocation and rank-zero failure order.
    dst = torch.empty(src.shape, dtype=torch.float32, device='cuda')
    src.shape[0]
    if not isinstance(BLOCK_SIZE, int):
        raise TypeError("BLOCK_SIZE must be an integer")
    if BLOCK_SIZE not in tuple(1 << k for k in range(21)):
        raise ValueError("BLOCK_SIZE must be a power of two in [1, 2**20]")
    if src.layout != torch.strided:
        raise TypeError("src must have strided layout")
    if src.dtype != torch.float32:
        raise TypeError("src must have dtype torch.float32")
    if not src.is_cuda:
        raise ValueError("src must be a CUDA tensor")
    if not src.is_contiguous():
        raise ValueError("src must be contiguous")
    if src.is_neg():
        raise ValueError("src must have no unresolved negative view bit")
    n_elements = src.numel()
    if triton.cdiv(n_elements, BLOCK_SIZE) > 2**31 - 1:
        raise ValueError("the one-dimensional grid exceeds the CUDA limit")
    if n_elements == 0:
        return dst
    if src.device != dst.device:
        raise ValueError("nonempty src must be on the current output device")
    if src.data_ptr() % 4 != 0:
        raise ValueError("src must have a 4-byte aligned data pointer")
    with torch.cuda.device(dst.device):
        mul_kernel_repaired[(triton.cdiv(n_elements, BLOCK_SIZE),)](
            src, dst, n_elements, BLOCK_SIZE)
    return dst

import triton
import triton.language as tl
import torch

@triton.jit
def sin_kernel(
    in_ptr0: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def sin_kernel_index_commuted(
    in_ptr0: tl.tensor,
    out_ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = BLOCK_SIZE * pid
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    output = tl.sin(x)
    tl.store(out_ptr + offsets, output, mask=mask)

def sin_triton_repaired(x, out):
    """Mutate the first x.numel() logical output cells; return None."""
    if not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor):
        raise TypeError("x and out must be tensors")
    if x.layout != torch.strided or out.layout != torch.strided:
        raise TypeError("x and out must have strided layout")
    if x.dtype != torch.float32 or out.dtype != torch.float32:
        raise TypeError("x and out must have dtype torch.float32")
    if not x.is_cuda or not out.is_cuda:
        raise ValueError("x and out must be CUDA tensors")
    if x.device != out.device:
        raise ValueError("x and out must be on the same device")
    n_elements = x.numel()
    if out.numel() < n_elements:
        raise ValueError("out must have at least x.numel() elements")
    if not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("x and out must be contiguous")
    if x.is_neg() or out.is_neg():
        raise ValueError("x and out must have no unresolved negative view bit")
    if n_elements >= 2**31:
        raise ValueError("x must contain fewer than 2**31 elements")
    if n_elements == 0:
        return None
    xp, op = x.data_ptr(), out.data_ptr()
    if xp % 4 != 0 or op % 4 != 0:
        raise ValueError("x and out must have 4-byte aligned data pointers")
    if xp != op and xp + 4*n_elements > op and op + 4*n_elements > xp:
        raise ValueError("active spans must be identical or disjoint")
    with torch.cuda.device(x.device):
        sin_kernel[((n_elements + 3) // 4,)](x, out, n_elements, BLOCK_SIZE=4)
    return None

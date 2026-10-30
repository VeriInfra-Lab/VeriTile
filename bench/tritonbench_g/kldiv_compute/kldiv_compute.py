"""Checked contiguous FP32 pointwise x * log(x / y), without input copies."""
import torch
import triton
import triton.language as tl

@triton.jit
def kldivergence_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)


def _check_inputs(x, y):
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("x and y must be tensors")
    if x.layout != torch.strided or y.layout != torch.strided:
        raise TypeError("x and y must have strided layout")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("x and y must have dtype torch.float32")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("x and y must be CUDA tensors")
    if x.device != y.device:
        raise ValueError("x and y must be on the same device")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    if x.is_neg() or y.is_neg():
        raise ValueError("x and y must not have unresolved negative view bits")

    n_elements = x.numel()
    if n_elements >= (1 << 31):
        raise ValueError("numel must be smaller than 2**31")
    # Scalar .b32 access: PTX ISA 8.7 section 6.4.1; see ../add_example/LOAD_STORE_AUDIT.md.
    if n_elements > 0 and (x.data_ptr() % 4 != 0 or y.data_ptr() % 4 != 0):
        raise ValueError("nonempty inputs must have 4-byte-aligned data pointers")

    return n_elements


def kldivergence(x, y):
    n_elements = _check_inputs(x, y)
    with torch.cuda.device(x.device):
        output = torch.empty_like(x)
        if n_elements:
            kldivergence_kernel[((n_elements + 1023) // 1024,)](
                x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

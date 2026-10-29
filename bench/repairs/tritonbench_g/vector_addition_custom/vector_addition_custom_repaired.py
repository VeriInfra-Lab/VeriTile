"""Checked FP32 CUDA vector addition preserving the original BLOCK16 kernel.

Inputs are same-shaped/device contiguous vectors with no unresolved negative
view bit, length below 2**31, and 4-byte-aligned nonempty data pointers.
The inputs are passed directly, including contiguous offset and aliased views.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(A: tl.tensor, B: tl.tensor, C: tl.tensor, size: tl.int32, BLOCK: tl.constexpr):
    """add kernel."""
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < size)
    b = tl.load(B + offs, mask=offs < size)
    tl.store(C + offs, a + b, mask=offs < size)


@triton.jit
def _add_commuted_kernel(A: tl.tensor, B: tl.tensor, C: tl.tensor, size: tl.int32, BLOCK: tl.constexpr):
    """add kernel."""
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < size)
    b = tl.load(B + offs, mask=offs < size)
    tl.store(C + offs, b + a, mask=offs < size)


def custom_add_repaired(x, y):
    """Return a fresh vector under Contract.lean's exact InputContract."""
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
    if x.ndim != 1:
        raise ValueError("x and y must be one-dimensional vectors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    if x.is_neg() or y.is_neg():
        raise ValueError("x and y must not have unresolved negative view bits")

    size = x.numel()
    if size >= (1 << 31):
        raise ValueError("numel must be smaller than 2**31")
    # Scalar .b32 access: PTX ISA 8.7 section 6.4.1; see ../add_example/LOAD_STORE_AUDIT.md.
    if size > 0 and (x.data_ptr() % 4 != 0 or y.data_ptr() % 4 != 0):
        raise ValueError("nonempty inputs must have 4-byte-aligned data pointers")


    with torch.cuda.device(x.device):
        c = torch.empty_like(x)
        if size == 0:
            return c
        BLOCK = 16
        grid = (triton.cdiv(size, BLOCK),)
        _add_kernel[grid](x, y, c, size, BLOCK=BLOCK)
        return c

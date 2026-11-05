"""FP32 addition with an explicit contiguous-input contract.

The original kernel is unchanged. Inputs must have the same shape and CUDA
device, use contiguous strided storage, have no unresolved negative view bit,
and contain fewer than 2**31 elements. Nonempty inputs must have 4-byte-aligned
data pointers. No input materialization is performed.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr: tl.tensor,  # *Pointer* to first input vector.
               y_ptr: tl.tensor,  # *Pointer* to second input vector.
               output_ptr: tl.tensor,  # *Pointer* to output vector.
               n_elements: tl.int32,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def add_commuted_kernel(x_ptr: tl.tensor,  # *Pointer* to first input vector.
               y_ptr: tl.tensor,  # *Pointer* to second input vector.
               output_ptr: tl.tensor,  # *Pointer* to output vector.
               n_elements: tl.int32,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = y + x
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    """Return a fresh FP32 sum under Contract.lean's InputContract.

    These ten checks match checkInputs/checkTensorInputs in the same order.
    Nonfinite FP32 values are admitted; no numerical assumption adds an input guard.
    """
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
    # Natural alignment for FP32 element accesses (PTX ISA 8.7, section 6.4.1).
    if n_elements > 0 and (x.data_ptr() % 4 != 0 or y.data_ptr() % 4 != 0):
        raise ValueError("nonempty inputs must have 4-byte-aligned data pointers")

    with torch.cuda.device(x.device):
        out = torch.empty_like(x)
        if n_elements == 0:
            return out

        BLOCK_SIZE = 1024
        num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        add_kernel[(num_blocks,)](x, y, out, n_elements, BLOCK_SIZE)
        return out

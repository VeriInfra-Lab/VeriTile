"""Elementwise cosine for contiguous FP32 CUDA tensors."""

import torch
import triton
import triton.language as tl


@triton.jit
def cos_func(a: tl.tensor, b: tl.tensor, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)



@triton.jit
def cos_func_index_commuted(a: tl.tensor, b: tl.tensor, n_elements: tl.int32, BLOCK_SIZE: tl.constexpr):
    offset = BLOCK_SIZE * tl.program_id(0) + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_value = tl.load(a + offset, mask=mask)
    b_value = tl.cos(a_value.to(tl.float32))
    tl.store(b + offset, b_value, mask=mask)

def cos_repaired(A):
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a Tensor")
    if A.layout != torch.strided:
        raise ValueError("A must use strided layout")
    if A.dtype != torch.float32:
        raise TypeError("A must have dtype float32")
    if not A.is_cuda:
        raise ValueError("A must be a CUDA tensor")
    if not A.is_contiguous():
        raise ValueError("A must be contiguous")
    if A.is_neg():
        raise ValueError("A must have no unresolved negative view bit")
    n_elements = A.numel()
    if n_elements >= 2**31:
        raise ValueError("A must contain fewer than 2**31 elements")
    if n_elements > 0 and A.data_ptr() % 4 != 0:
        raise ValueError("A must have a 4-byte-aligned data pointer")
    B = torch.empty_like(A)
    if n_elements == 0:
        return B
    block_size = 256
    grid_size = triton.cdiv(n_elements, block_size)
    with torch.cuda.device(A.device):
        cos_func[(grid_size, 1, 1)](A, B, n_elements, block_size)
    return B

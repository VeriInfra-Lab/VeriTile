"""Logical-view repair of kldiv_compute; original FP expression/order retained."""
import torch
import triton
import triton.language as tl


@triton.jit
def kldivergence_kernel(x_ptr, y_ptr, output_ptr, n_elements: tl.int64,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x * tl.log(x / y)
    tl.store(output_ptr + offsets, output, mask=mask)


def kldivergence(x, y):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    if y.numel() < x.numel():
        raise ValueError("kldivergence requires at least x.numel() logical y elements")
    if x.device != y.device:
        raise ValueError("kldivergence requires the same CUDA device")
    packed_x = x.resolve_neg().clone(memory_format=torch.contiguous_format)
    packed_y = y.resolve_neg().clone(memory_format=torch.contiguous_format)
    work = torch.empty_like(packed_x, memory_format=torch.contiguous_format)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    kldivergence_kernel[grid](packed_x, packed_y, work, n_elements, BLOCK_SIZE=1024)
    output.copy_(work)
    return output

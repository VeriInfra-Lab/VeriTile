"""FP32 contiguous doubling and MUL-COMMUTE candidates; guards specify inputs."""
import torch
import triton
from triton import language as tl


@triton.jit
def mul2_kernel(
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
    output = 2 * x
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def mul2_inplace_kernel(
    ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(ptr + offsets, mask=mask)
    output = 2 * x
    tl.store(ptr + offsets, output, mask=mask)


# Candidates used by the kernel equivalence proof.
@triton.jit
def mul2_commuted_kernel(
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
    output = x * 2
    tl.store(out_ptr + offsets, output, mask=mask)


@triton.jit
def mul2_inplace_commuted_kernel(
    ptr: tl.tensor,
    n_elements: tl.int32,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(ptr + offsets, mask=mask)
    output = x * 2
    tl.store(ptr + offsets, output, mask=mask)


def _check(x, BLOCK_SIZE):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.layout != torch.strided:
        raise ValueError("x must have strided layout")
    if x.dtype != torch.float32:
        raise ValueError("x must be FP32")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.is_neg():
        raise ValueError("x must have no unresolved negative view bit")
    if x.numel() >= 2**31:
        raise ValueError("x must contain fewer than 2**31 elements")
    if x.numel() > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("x must have an FP32-aligned data pointer")
    if type(BLOCK_SIZE) is not int or BLOCK_SIZE not in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576):
        raise ValueError("BLOCK_SIZE must be a power of two from 1 through 2**20")


def triton_mul2(x, BLOCK_SIZE=16):
    _check(x, BLOCK_SIZE)
    output = torch.zeros_like(x)
    n_elements = x.numel()
    if n_elements:
        with torch.cuda.device(x.device):
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            mul2_kernel[grid](x, output, n_elements, BLOCK_SIZE)
    return output


def triton_mul2_inplace(x, BLOCK_SIZE=16):
    _check(x, BLOCK_SIZE)
    n_elements = x.numel()
    if n_elements:
        with torch.cuda.device(x.device):
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            mul2_inplace_kernel[grid](x, n_elements, BLOCK_SIZE)
    return x


import math
import torch
import triton
import triton.language as tl


# TODO: autotune this better.
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    block_start = pid * tl.full((), BLOCK_SIZE, tl.int64)
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    safe_max = tl.where(max_val == 0.0, 1.0, max_val)
    output = tl.extra.cuda.libdevice.llrint(127.0 * (x / safe_max))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def _check(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a Tensor")
    if x.layout != torch.strided or x.dtype != torch.float32:
        raise ValueError("x must be strided FP32")
    if not x.is_cuda:
        raise ValueError("x must be on CUDA")
    if x.ndim != 2:
        raise ValueError("x must have rank two")
    if not x.is_contiguous() or x.is_neg():
        raise ValueError("x must be contiguous with no unresolved negative bit")
    if not 0 < x.shape[1] <= 2**20:
        raise ValueError("column count must be from one through 2**20")
    if x.shape[0] >= 2**31:
        raise ValueError("row grid must fit signed int32")
    if x.numel() > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("x must have an FP32-aligned data pointer")


def quantize_rowwise(x: torch.Tensor):
    _check(x)
    with torch.cuda.device(x.device):
        output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)
        if x.shape[0] == 0:
            return output, output_maxs
        P2 = triton.next_power_of_2(x.shape[1])
        n_elements = output.numel()
        grid = lambda meta: (x.shape[0],)
        _quantize_rowwise[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
        return output, output_maxs


# Store scheduling candidate: release max_val before quantization.
# TODO: autotune this better.
@triton.autotune(
    configs=[
        triton.Config({}, num_stages=1, num_warps=8),
        triton.Config({}, num_stages=2, num_warps=8),
        triton.Config({}, num_stages=4, num_warps=8),
        triton.Config({}, num_stages=8, num_warps=8),
        triton.Config({}, num_stages=1),
        triton.Config({}, num_stages=2),
        triton.Config({}, num_stages=4),
        triton.Config({}, num_stages=8),
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_rowwise_early_max(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    block_start = pid * tl.full((), BLOCK_SIZE, tl.int64)
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)

    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    tl.store(output_maxs + pid, max_val)
    safe_max = tl.where(max_val == 0.0, 1.0, max_val)
    output = tl.extra.cuda.libdevice.llrint(127.0 * (x / safe_max))
    tl.store(output_ptr + offsets, output, mask=row_mask)


def quantize_rowwise_early_max(x: torch.Tensor):
    _check(x)
    with torch.cuda.device(x.device):
        output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        output_maxs = torch.empty(x.shape[0], device=x.device, dtype=torch.float16)
        if x.shape[0] == 0:
            return output, output_maxs
        P2 = triton.next_power_of_2(x.shape[1])
        n_elements = output.numel()
        grid = lambda meta: (x.shape[0],)
        _quantize_rowwise_early_max[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
        return output, output_maxs

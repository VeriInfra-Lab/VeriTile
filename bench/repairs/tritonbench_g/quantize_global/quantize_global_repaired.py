"""Separate FP32 global quantizer; the original corpus source is unchanged.

For finite inputs this quantizes the logical tensor view using one global
absolute maximum. Empty inputs return an empty int8 tensor and absmax [0].
NaN/Inf, non-FP32, non-strided and non-CUDA inputs are explicitly rejected.
The repaired arithmetic uses div.rn.f32 then mul.f32, not the original
host reciprocal followed by two multiplications. No bitwise equivalence to
that defective numerical graph is claimed.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=1),
    ],
    key=["n_elements"],
)
@triton.jit
def _quantize_global_repaired(
    x_ptr,
    absmax_ptr,
    output_ptr,
    n_elements: tl.int64,
    base: tl.int64,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64) + base
    mask = offsets < n_elements
    safe_offsets = tl.where(mask, offsets, 0)
    x = tl.load(x_ptr + safe_offsets, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.load(absmax_ptr).to(tl.float32)
    denominator = tl.where(absmax == 0.0, 1.0, absmax)
    normalized = tl.div_rn(x, denominator)
    scaled = 127.0 * normalized
    lower_bounded = tl.where(scaled < -127.0, -127.0, scaled)
    bounded = tl.where(lower_bounded > 127.0, 127.0, lower_bounded)
    output = tl.extra.cuda.libdevice.llrint(bounded)
    tl.store(output_ptr + safe_offsets, output, mask=mask)


def quantize_global(x: torch.Tensor):
    if not isinstance(x, torch.Tensor):
        raise TypeError("quantize_global requires a torch.Tensor")
    if x.layout != torch.strided or x.dtype != torch.float32:
        raise TypeError("quantize_global requires a strided FP32 tensor")
    if not x.is_cuda:
        raise ValueError("quantize_global requires a CUDA tensor")
    if not x.is_contiguous() or x.is_neg():
        raise ValueError("quantize_global requires contiguous input with no unresolved negative bit")
    if x.numel() >= 2**63:
        raise ValueError("element count must fit signed int64")
    if x.numel() > 0 and x.data_ptr() % 4 != 0:
        raise ValueError("input data pointer must be FP32 aligned")
    logical = x
    with torch.cuda.device(x.device):
        output = torch.empty(x.shape, device=x.device, dtype=torch.int8)
        n_elements = logical.numel()
        if n_elements == 0:
            return output, torch.zeros((1,), device=x.device, dtype=torch.float32)
        if not torch.isfinite(logical).all().item():
            raise ValueError("quantize_global requires finite FP32 values")
        absmax = logical.abs().amax().reshape(1)
        # Both autotune block sizes divide this span. Every launch has at most
        # 2**20 blocks; the final masked tail is the only partially covered tile.
        for base in range(0, n_elements, 1 << 30):
            count = min(n_elements - base, 1 << 30)
            grid = lambda meta: (triton.cdiv(count, meta["BLOCK_SIZE"]),)
            _quantize_global_repaired[grid](logical, absmax, output, n_elements, base)
        return output, absmax

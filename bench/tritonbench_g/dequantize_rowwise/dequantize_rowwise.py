
import math
import torch
import triton
import triton.language as tl

# Triton kernel for rowwise dequantization
@triton.jit
def _dequantize_rowwise_repaired(
    x_ptr,
    state_x,
    output_ptr,
    inv_127,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2).to(tl.int64)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)
    max_val = tl.load(state_x + pid)
    output = max_val * x * inv_127
    tl.store(output_ptr + offsets, output, mask=row_mask)

def _check(x, state_x):
    if not isinstance(x, torch.Tensor) or not isinstance(state_x, torch.Tensor):
        raise TypeError("dequantize_rowwise requires two Tensors")
    if x.layout != torch.strided or state_x.layout != torch.strided:
        raise TypeError("both tensors must use strided layout")
    if x.dtype != torch.int8 or state_x.dtype != torch.float32:
        raise TypeError("x must be INT8 and scales must be FP32")
    if not x.is_cuda or not state_x.is_cuda:
        raise ValueError("both tensors must be CUDA")
    if x.device != state_x.device:
        raise ValueError("both tensors must be on the same device")
    if x.ndim != 2 or state_x.ndim != 1:
        raise ValueError("input must be rank two and scales rank one")
    if not x.is_contiguous() or not state_x.is_contiguous() or x.is_neg() or state_x.is_neg():
        raise ValueError("inputs must be contiguous with no unresolved negative bit")
    if x.shape[1] > 2**20:
        raise ValueError("row width exceeds Triton block limit")
    if x.shape[0] >= 2**31:
        raise ValueError("row grid must fit signed int32")
    if x.numel() > 0 and x.shape[0] > state_x.numel():
        raise ValueError("nonempty input requires one scale per row")
    if x.numel() > 0 and state_x.data_ptr() % 4 != 0:
        raise ValueError("scale pointer must be FP32 aligned")


def dequantize_rowwise_repaired(x: torch.Tensor, state_x: torch.Tensor):
    _check(x, state_x)
    with torch.cuda.device(x.device):
        output = torch.empty(x.shape, device=x.device, dtype=torch.float16)
        if output.numel() == 0:
            return output
        P2 = triton.next_power_of_2(x.shape[1])
        n_elements = output.numel()
        grid = lambda meta: (x.shape[0],)
        _dequantize_rowwise_repaired[grid](x, state_x, output, 1.0 / 127, n_elements,
                                         BLOCK_SIZE=x.shape[1], P2=P2)
        return output

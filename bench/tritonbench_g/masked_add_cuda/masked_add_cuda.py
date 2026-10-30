"""Contiguous FP32 masked in-place add; the returned Python value is None."""
import torch
import triton
import triton.language as tl


@triton.jit
def masked_add_kernel(grad_ptr, p_ptr, p_mask_ptr, n_elements, alpha,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad += p * alpha
    tl.store(grad_ptr + offsets, grad, mask=mask)


def masked_add(grad: torch.Tensor, p_data: torch.Tensor,
               p_mask: torch.Tensor, alpha: float = 0):
    """Update grad where the signed integer mask is zero; return None.

    Inputs have equal shapes/devices, contiguous layouts, and no negative view
    bits. Grad/data are FP32; mask is int32 or int64. Grad's live byte interval
    must be disjoint from both input intervals. Alpha is a Python float or the
    default integer zero. Empty inputs perform no pointer queries or launch.
    """
    tensors = (grad, p_data, p_mask)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("masked_add requires Tensor inputs")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("masked_add requires strided tensors")
    if grad.dtype != torch.float32 or p_data.dtype != torch.float32 or p_mask.dtype not in (torch.int32, torch.int64):
        raise TypeError("masked_add requires FP32 grad/data and signed int32/int64 mask")
    if not all(t.is_cuda for t in tensors):
        raise ValueError("masked_add requires CUDA tensors")
    if grad.device != p_data.device or grad.device != p_mask.device:
        raise ValueError("masked_add requires one CUDA device")
    if grad.shape != p_data.shape or grad.shape != p_mask.shape:
        raise ValueError("masked_add requires equal shapes")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("masked_add requires contiguous tensors")
    if any(t.is_neg() for t in tensors):
        raise ValueError("masked_add does not accept negative views")
    if not (type(alpha) is float or (type(alpha) is int and alpha == 0)):
        raise TypeError("alpha must be a Python float or integer zero")
    n_elements = grad.numel()
    if n_elements >= 2**31:
        raise ValueError("masked_add requires fewer than 2**31 elements")
    if n_elements == 0:
        return None
    gptr, pptr, mptr = (t.data_ptr() for t in tensors)
    mask_bytes = p_mask.element_size()
    if gptr % 4 or pptr % 4 or mptr % mask_bytes:
        raise ValueError("masked_add requires naturally aligned pointers")
    grad_end = gptr + 4 * n_elements
    if not ((grad_end <= pptr or pptr + 4 * n_elements <= gptr) and
            (grad_end <= mptr or mptr + mask_bytes * n_elements <= gptr)):
        raise ValueError("masked_add requires grad disjoint from data and mask")
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch.cuda.device(grad.device):
        masked_add_kernel[grid](grad, p_data, p_mask, n_elements, float(alpha),
                                BLOCK_SIZE=1024, enable_fp_fusion=True)
    return None

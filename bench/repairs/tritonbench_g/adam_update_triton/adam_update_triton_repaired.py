"""Separate default-FP32 repair of the source-named Adam (actually Lion).
Original SHA256: 27d4f15342638339b2daea53f43752e30a28114a39d1b8dcdaf5e29b89072f45
"""

import torch
import triton
import triton.language as tl

@triton.autotune(configs = [
    triton.Config({'BLOCK_SIZE': 128}, num_warps = 4),
    triton.Config({'BLOCK_SIZE': 1024}, num_warps = 8),
], key = ['n_elements'], restore_value=['p_ptr', 'exp_avg_ptr'])

# Triton CUDA kernel

@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    lr,
    wd,
    beta1,
    beta2,
    n_elements,
    chunk_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < chunk_elements

    # Offsetted pointers
    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets

    # Load
    p = tl.load(offset_p_ptr, mask=mask)
    grad = tl.load(offset_grad_ptr, mask=mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

    # Stepweight decay
    p = p * (1 - lr * wd)

    # Diff between momentum running average and grad
    diff = exp_avg - grad

    # Weight update
    update = diff * beta1 + grad

    # torch.sign
    can_update = update != 0
    update_sign = tl.where(update > 0, -lr, lr)

    p = p + update_sign * can_update

    # Decay the momentum running average coefficient
    exp_avg = diff * beta2 + grad

    # Store new params and momentum running average coefficient
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)


def _numeric_scalar(value, name):
    """Normalize supported numeric payloads to the runtime FP32 scalar ABI.

    Python binary64 conversion precedes Triton's FP32 conversion. Built-in
    conversion slots retain numeric subclasses without running their overrides.
    Tensor scalars and arbitrary coercible objects are not numeric payloads.
    """
    if type(value) is tl.constexpr:
        value = value.value
    if issubclass(type(value), int):
        return int.__float__(value)
    if issubclass(type(value), float):
        return float.__float__(value)
    raise TypeError(f"{name} requires a bool, int, float or numeric tl.constexpr")


def update_fn_repaired(p, grad, exp_avg, lr, wd, beta1, beta2):
    """In-place FP32 Lion update on checked direct views; returns None."""
    tensors = (p, grad, exp_avg)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("p, grad and exp_avg must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided tensors required")
    if any(t.dtype != torch.float32 for t in tensors):
        raise TypeError("selected Lion API requires FP32 payloads")
    if not all(t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if grad.device != p.device or exp_avg.device != p.device:
        raise ValueError("all tensors must use the same CUDA device")
    n_elements = p.numel()
    if grad.numel() < n_elements or exp_avg.numel() < n_elements:
        raise ValueError("gradient and momentum need the parameter logical prefix")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views are unsupported")
    if n_elements == 0:
        return None
    if any(t.data_ptr() % 16 != 0 for t in tensors):
        raise ValueError("nonempty inputs must select the reviewed 16-byte-aligned profile")
    ranges = [(t.data_ptr(), t.data_ptr() + 4 * t.numel()) for t in tensors]
    if any(a[0] < b[1] and b[0] < a[1]
           for i, a in enumerate(ranges) for b in ranges[i + 1:]):
        raise ValueError("parameter, gradient and momentum byte ranges must be disjoint")
    lr = _numeric_scalar(lr, "lr")
    wd = _numeric_scalar(wd, "wd")
    beta1 = _numeric_scalar(beta1, "beta1")
    beta2 = _numeric_scalar(beta2, "beta2")
    with torch.no_grad(), torch.cuda.device(p.device):
        p_flat = p.view(-1)
        grad_flat = grad.view(-1)
        exp_avg_flat = exp_avg.view(-1)
        for first in range(0, n_elements, 65535 * 128):
            last = min(n_elements, first + 65535 * 128)
            chunk_elements = last - first
            grid = lambda meta: (triton.cdiv(chunk_elements, meta['BLOCK_SIZE']),)
            update_fn_kernel[grid](
                p_flat[first:last], grad_flat[first:last], exp_avg_flat[first:last],
                lr, wd, beta1, beta2, n_elements, chunk_elements,
            )
    return None

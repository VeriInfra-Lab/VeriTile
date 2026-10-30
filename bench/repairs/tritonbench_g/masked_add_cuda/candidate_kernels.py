"""Existing addition-order candidate for the selected repaired source.

Both signed mask widths use this same unchanged function body. The launch
retains enable_fp_fusion=True, interpreted by the selected native graph.
"""
import triton
import triton.language as tl


@triton.jit
def commuted_masked_add(grad_ptr, p_ptr, p_mask_ptr, n_elements, alpha,
                        BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p_mask = tl.load(p_mask_ptr + offsets, mask=mask).to(tl.int1)
    mask = mask & ~p_mask
    p = tl.load(p_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    grad = p * alpha + grad
    tl.store(grad_ptr + offsets, grad, mask=mask)

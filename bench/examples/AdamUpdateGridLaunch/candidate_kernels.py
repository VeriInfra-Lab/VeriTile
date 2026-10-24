"""Existing candidate: commute only the final momentum addition in the
original Adam/Lion update source. Runtime ABI, masks and both in-place stores
are unchanged. The separate arithmetic graph uses enable_fp_fusion=False.
"""
import triton
import triton.language as tl

@triton.jit
def commuted_update(p_ptr, grad_ptr, exp_avg_ptr, lr, wd, beta1, beta2, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

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
    exp_avg = grad + diff * beta2

    # Store new params and momentum running average coefficient
    tl.store(offset_p_ptr, p, mask=mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)

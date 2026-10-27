"""Python spelling of the existing Lean output-row integer-commutation candidates.

Derived from the unchanged selected repaired kernels. The only body changes
commute the output pointer's row/stride product; floating expressions, masks,
casts, flags, reduction modes and loop order are identical.
"""
import triton
import triton.language as tl

_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

@triton.jit
def _kldiv_kernel_forward_index_commuted(
    y_ptr,  # [B, S], prediction ptr, the kernel expects the prediction in log-space
    y_stride,  # int, prediction stride
    gt_ptr,  # [B, S], ground truth ptr
    gt_stride,  # int, ground truth stride
    loss_ptr,  # [B] or [B, S] if reduction == _REDUCTION_MODE_NONE, output ptr
    loss_stride,  # int, output stride
    n_cols,  # int, number of columns in the input tensor
    eps,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    n_cols = tl.cast(n_cols, tl.int64)
    y_stride = tl.cast(y_stride, tl.int64)
    gt_stride = tl.cast(gt_stride, tl.int64)
    loss_stride = tl.cast(loss_stride, tl.int64)
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += loss_stride * pid

    base_offsets = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)

        # KL(y_true || y) = y_true * (log(y_true) - log(y))
        # We compute KL(y_true || y) with y in the log-space
        if not log_target:
            loss = y_true * (tl.log(tl.maximum(y_true, eps)) - y)
        else:
            loss = tl.exp(y_true) * (y_true - y)

        if reduction == 0:  # _REDUCTION_MODE_NONE
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.where(mask, loss, 0.0)
            loss_sum += tl.sum(loss, axis=0)

    if reduction != 0:
        tl.store(loss_ptr, loss_sum)

@triton.jit
def _kldiv_kernel_backward_index_commuted(
    target_ptr,
    target_stride,
    new_grads_ptr,
    new_grads_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    n_cols = tl.cast(n_cols, tl.int64)
    target_stride = tl.cast(target_stride, tl.int64)
    new_grads_stride = tl.cast(new_grads_stride, tl.int64)
    pid = tl.program_id(0).to(tl.int64)

    target_ptr += pid * target_stride
    new_grads_ptr += new_grads_stride * pid

    offsets = tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
    mask = offsets < n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.cast(tl.arange(0, BLOCK_SIZE), tl.int64)
        mask = offsets < n_cols

        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)

        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)

        tl.store(new_grads_ptr + offsets, res, mask=mask)

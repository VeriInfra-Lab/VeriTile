"""Newly authored Python for the existing Lean output-address candidates.

The historical original remains kldiv_triton_repaired.py. This file is its
retained candidate's Python counterpart, not an invented original reference.
"""
import triton
import triton.language as tl
_REDUCTION_MODE_NONE = tl.constexpr(0)
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

@triton.jit
def _kldiv_kernel_forward_index_commuted(
    y_ptr,
    y_stride: tl.int64,
    gt_ptr,
    gt_stride: tl.int64,
    loss_ptr,
    loss_stride: tl.int64,
    n_cols: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    y_ptr += pid * y_stride
    gt_ptr += pid * gt_stride
    loss_ptr += loss_stride * pid
    base_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        y_true = tl.load(gt_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            loss = y_true * (tl.log(y_true) - y)
            loss = tl.where(y_true == 0.0, 0.0, loss)
        else:
            loss = tl.exp(y_true) * (y_true - y)
        loss = tl.where(mask, loss, 0.0)
        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, loss, mask=mask)
        else:
            loss = tl.sum(loss, axis=0)
            tl.store(loss_ptr, loss)
            loss_ptr += 1

@triton.jit
def _kldiv_kernel_backward_index_commuted(
    input_ptr,
    input_stride: tl.int64,
    target_ptr,
    target_stride: tl.int64,
    n_cols: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    input_ptr += input_stride * pid
    target_ptr += pid * target_stride
    offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offsets < n_cols
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = offsets < n_cols
        target = tl.load(target_ptr + offsets, mask=mask, other=0.0)
        if not log_target:
            res = target * -1
        else:
            res = -tl.exp(target)
        tl.store(input_ptr + offsets, res, mask=mask)


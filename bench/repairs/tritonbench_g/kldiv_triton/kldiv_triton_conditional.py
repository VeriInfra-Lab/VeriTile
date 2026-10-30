import math

import torch
import triton
import triton.language as tl


def get_num_warps(BLOCK_SIZE):
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return num_warps


MAX_FUSED_SIZE = 65536 // 4
_REDUCTION_MODE_NONE = tl.constexpr(0)
_REDUCTION_MODE_SUM = tl.constexpr(1)
_REDUCTION_MODE_MEAN = tl.constexpr(2)
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)
_str_to_reduction_mode = {
    "none": _REDUCTION_MODE_NONE.value,
    "sum": _REDUCTION_MODE_SUM.value,
    "mean": _REDUCTION_MODE_MEAN.value,
    "batchmean": _REDUCTION_MODE_BATCHMEAN.value,
}


@triton.jit(
    do_not_specialize=["y_ptr", "y_stride", "gt_ptr", "gt_stride", "loss_ptr", "loss_stride", "n_cols"],
    do_not_specialize_on_alignment=["y_ptr", "y_stride", "gt_ptr", "gt_stride", "loss_ptr", "loss_stride", "n_cols"],
)
def _kldiv_kernel_forward_repaired(
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
    loss_ptr += pid * loss_stride
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


@triton.jit(
    do_not_specialize=["input_ptr", "input_stride", "target_ptr", "target_stride", "n_cols"],
    do_not_specialize_on_alignment=["input_ptr", "input_stride", "target_ptr", "target_stride", "n_cols"],
)
def _kldiv_kernel_backward_repaired(
    input_ptr,
    input_stride: tl.int64,
    target_ptr,
    target_stride: tl.int64,
    n_cols: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    log_target: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    input_ptr += pid * input_stride
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


def _check_writable_matrix(tensor, rows, cols):
    """Reject distinct logical outputs that occupy the same physical word."""
    if rows == 0 or cols == 0:
        return
    row_stride, col_stride = tensor.stride()
    if (rows > 1 and row_stride == 0) or (cols > 1 and col_stride == 0):
        raise ValueError("kldiv backward destination has overlapping logical elements")
    if rows > 1 and cols > 1:
        divisor = math.gcd(row_stride, col_stride)
        if col_stride // divisor < rows and row_stride // divisor < cols:
            raise ValueError("kldiv backward destination has overlapping logical elements")


def kldiv_forward_triton(y_pred, y_true, log_target, reduction):
    B, S = y_pred.shape
    BLOCK_SIZE = max(1, min(MAX_FUSED_SIZE, triton.next_power_of_2(S)))
    num_warps = get_num_warps(BLOCK_SIZE)
    grid = (B,)
    reduction = _str_to_reduction_mode[reduction]
    if y_true.shape != y_pred.shape:
        raise ValueError("kldiv forward requires matching prediction and target shapes")
    if y_true.device != y_pred.device:
        raise ValueError("kldiv forward requires matching input devices")
    chunks = triton.cdiv(S, BLOCK_SIZE)
    out_size = (B, S) if reduction == _REDUCTION_MODE_NONE.value else (B, chunks)
    output_tensor = torch.zeros(out_size, dtype=torch.float32, device=y_pred.device)
    with torch.cuda.device(y_pred.device):
        prediction = y_pred.resolve_neg().contiguous()
        target = y_true.resolve_neg().contiguous()
        _kldiv_kernel_forward_repaired[grid](
            prediction,
            prediction.stride(0),
            target,
            target.stride(0),
            output_tensor,
            output_tensor.stride(0),
            S,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            log_target=log_target,
            reduction=reduction,
        )
        if reduction == _REDUCTION_MODE_BATCHMEAN.value:
            return output_tensor.sum() / B
        elif reduction == _REDUCTION_MODE_SUM.value:
            return output_tensor.sum()
        elif reduction == _REDUCTION_MODE_MEAN.value:
            return output_tensor.sum() / (B * S)
        else:
            return output_tensor


def kldiv_backward_triton(input, target, grad_output, log_target):
    B, S = input.shape
    BLOCK_SIZE = max(1, min(MAX_FUSED_SIZE, triton.next_power_of_2(S)))
    num_warps = get_num_warps(BLOCK_SIZE)
    grid = (B,)
    if target.shape != input.shape:
        raise ValueError("kldiv backward requires matching destination and target shapes")
    if target.device != input.device:
        raise ValueError("kldiv backward requires matching destination and target devices")
    _check_writable_matrix(input, B, S)
    with torch.cuda.device(input.device):
        target_values = target.resolve_neg().clone(memory_format=torch.contiguous_format)
        gradient = grad_output.resolve_neg().clone(memory_format=torch.contiguous_format)
        work = torch.empty((B, S), dtype=input.dtype, device=input.device)
        _kldiv_kernel_backward_repaired[grid](
            work,
            work.stride(0),
            target_values,
            target_values.stride(0),
            S,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            log_target=log_target,
        )
        with torch.no_grad():
            input.copy_(work)
        if torch.equal(gradient, torch.tensor(1.0, dtype=gradient.dtype, device=gradient.device)):
            return input
        return input * gradient

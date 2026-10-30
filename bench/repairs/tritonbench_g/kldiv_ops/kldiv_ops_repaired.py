from typing import Literal
import torch
import triton
import triton.language as tl


MAX_FUSED_SIZE = 65536 // 4  # 65536 // 4 or 8 works the best
REDUCTION_LITERAL = Literal["none", "sum", "mean", "batchmean"]
_REDUCTION_MODE_BATCHMEAN = tl.constexpr(3)

@triton.jit
def _kldiv_kernel_forward(
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
    loss_ptr += pid * loss_stride

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
def _kldiv_kernel_backward(
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
    new_grads_ptr += pid * new_grads_stride

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


def _check_tensors(*tensors):
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("kldiv requires Tensor arguments")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("kldiv requires strided tensors")
    if any(t.dtype != torch.float32 for t in tensors):
        raise ValueError("kldiv requires FP32 tensors")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("kldiv requires CUDA tensors")
    if any(t.device != tensors[0].device for t in tensors[1:]):
        raise ValueError("kldiv requires tensors on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("kldiv requires contiguous tensors")
    if any(t.is_neg() for t in tensors):
        raise ValueError("kldiv requires resolved negative views")
    if any(t.numel() != 0 and t.data_ptr() % 4 != 0 for t in tensors):
        raise ValueError("kldiv requires aligned active FP32 pointers")


def _check_matrix_pair(a, b):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("kldiv requires rank-two matrices")
    if a.shape != b.shape:
        raise ValueError("kldiv requires matching matrix shapes")


def _storage_separate(a, b):
    a_storage, b_storage = a.untyped_storage(), b.untyped_storage()
    a_base, b_base = a_storage.data_ptr(), b_storage.data_ptr()
    return (a_base + a_storage.nbytes() <= b_base or
            b_base + b_storage.nbytes() <= a_base)


def kldiv_forward_triton(y_pred, y_true, log_target, reduction, eps):
    """Checked rank-two contiguous FP32 CUDA clamped KL divergence."""
    _check_tensors(y_pred, y_true)
    _check_matrix_pair(y_pred, y_true)
    if type(log_target) is not bool:
        raise TypeError("log_target must be bool")
    if not isinstance(reduction, str) or reduction not in ("none", "sum", "mean", "batchmean"):
        raise ValueError("unsupported KL reduction")
    if type(eps) is not float:
        raise TypeError("eps must be a Python float")
    BT, V = y_pred.shape
    BLOCK_SIZE = min(16384, triton.next_power_of_2(max(1, V)))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32
    mode = {"none": 0, "sum": 1, "mean": 2, "batchmean": 3}[reduction]
    out_size = (BT, V) if mode == 0 else (BT,)
    output_tensor = torch.zeros(out_size, device=y_pred.device, dtype=torch.float32)
    with torch.cuda.device(y_pred.device):
        _kldiv_kernel_forward[(BT,)](
            y_pred, V, y_true, V, output_tensor, V if mode == 0 else 1, V,
            eps=eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            log_target=log_target, reduction=mode,
        )
        if mode == 3:
            return output_tensor.sum() / BT
        if mode == 1:
            return output_tensor.sum(dim=0)
        if mode == 2:
            return output_tensor.sum() / (BT * V)
        return output_tensor


def kldiv_backward_triton(target, grad_output, new_grads, log_target):
    """Write the unscaled derivative directly, then apply the cotangent."""
    _check_tensors(new_grads, target, grad_output)
    _check_matrix_pair(new_grads, target)
    if grad_output.ndim != 0 and grad_output.shape != new_grads.shape:
        raise ValueError("grad_output must be scalar or have the matrix shape")
    if type(log_target) is not bool:
        raise TypeError("log_target must be bool")
    if new_grads.numel() != 0:
        if not _storage_separate(new_grads, target) or not _storage_separate(new_grads, grad_output):
            raise ValueError("destination storage must be separate from target and cotangent")
    BT, V = target.shape
    BLOCK_SIZE = min(16384, triton.next_power_of_2(max(1, V)))
    num_warps = 4 if BLOCK_SIZE < 2048 else 8 if BLOCK_SIZE < 8192 else 16 if BLOCK_SIZE < 32768 else 32
    with torch.cuda.device(new_grads.device):
        _kldiv_kernel_backward[(BT,)](
            target, V, new_grads, V, V,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, log_target=log_target,
        )
        if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
            return new_grads
        return new_grads * grad_output

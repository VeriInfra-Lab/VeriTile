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


def kldiv_forward_triton(y_pred, y_true, log_target, reduction):
    """Checked rank-two contiguous FP32 CUDA KL divergence."""
    _check_tensors(y_pred, y_true)
    _check_matrix_pair(y_pred, y_true)
    if type(log_target) is not bool:
        raise TypeError("log_target must be bool")
    if not isinstance(reduction, str) or reduction not in _str_to_reduction_mode:
        raise ValueError("unsupported KL reduction")
    B, S = y_pred.shape
    BLOCK_SIZE = max(1, min(MAX_FUSED_SIZE, triton.next_power_of_2(S)))
    num_warps = get_num_warps(BLOCK_SIZE)
    mode = _str_to_reduction_mode[reduction]
    chunks = triton.cdiv(S, BLOCK_SIZE)
    out_size = (B, S) if mode == _REDUCTION_MODE_NONE.value else (B, chunks)
    output_tensor = torch.zeros(out_size, dtype=torch.float32, device=y_pred.device)
    with torch.cuda.device(y_pred.device):
        _kldiv_kernel_forward_repaired[(B,)](
            y_pred, y_pred.stride(0), y_true, y_true.stride(0),
            output_tensor, output_tensor.stride(0), S,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
            log_target=log_target, reduction=mode,
        )
        if mode == _REDUCTION_MODE_BATCHMEAN.value:
            return output_tensor.sum() / B
        if mode == _REDUCTION_MODE_SUM.value:
            return output_tensor.sum()
        if mode == _REDUCTION_MODE_MEAN.value:
            return output_tensor.sum() / (B * S)
        return output_tensor


def kldiv_backward_triton(input, target, grad_output, log_target):
    """Write the unscaled derivative into input, then apply the cotangent."""
    _check_tensors(input, target, grad_output)
    _check_matrix_pair(input, target)
    if grad_output.ndim != 0 and grad_output.shape != input.shape:
        raise ValueError("grad_output must be scalar or have the matrix shape")
    if type(log_target) is not bool:
        raise TypeError("log_target must be bool")
    if input.numel() != 0:
        if not _storage_separate(input, target) or not _storage_separate(input, grad_output):
            raise ValueError("destination storage must be separate from target and cotangent")
    B, S = input.shape
    BLOCK_SIZE = max(1, min(MAX_FUSED_SIZE, triton.next_power_of_2(S)))
    num_warps = get_num_warps(BLOCK_SIZE)
    with torch.cuda.device(input.device):
        _kldiv_kernel_backward_repaired[(B,)](
            input, input.stride(0), target, target.stride(0), S,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, log_target=log_target,
        )
        if torch.equal(grad_output, torch.tensor(1.0, dtype=grad_output.dtype, device=grad_output.device)):
            return input
        return input * grad_output

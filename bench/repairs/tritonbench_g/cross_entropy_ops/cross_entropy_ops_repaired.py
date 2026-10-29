"""Checked single-process entry calling the exact preserved resident source."""
import importlib.util
import math
from pathlib import Path
import torch
import triton

_source_spec = importlib.util.spec_from_file_location(
    "_veritile_cross_entropy_ops_source", Path(__file__).with_name("cross_entropy_ops_conditional.py"))
_source = importlib.util.module_from_spec(_source_spec)
_source_spec.loader.exec_module(_source)

# The resident theorem and source pin apply to these original function objects.
cross_entropy_fwd_kernel = _source.cross_entropy_fwd_kernel
cross_entropy_bwd_kernel = _source.cross_entropy_bwd_kernel

def _check_forward(logits, labels, smoothing, logit_scale, square_scale, ignored, inplace, process_group):
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be Tensors")
    if logits.layout != torch.strided or labels.layout != torch.strided:
        raise TypeError("logits and labels must have strided layout")
    if logits.dtype != torch.float32 or labels.dtype != torch.int64:
        raise TypeError("expected FP32 logits and int64 labels")
    if not logits.is_cuda or not labels.is_cuda or logits.device != labels.device:
        raise ValueError("logits and labels must share one CUDA device")
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError("expected logits [rows, classes] and labels [rows]")
    if not logits.is_contiguous() or logits.stride(-1) != 1 or not labels.is_contiguous():
        raise ValueError("logits and labels must be contiguous with unit logit column stride")
    if logits.is_neg() or labels.is_neg():
        raise ValueError("inputs must have no unresolved negative view bit")
    if not isinstance(smoothing, float) or not isinstance(square_scale, float):
        raise TypeError("smoothing and lse_square_scale must be Python floats")
    if not math.isfinite(smoothing) or not 0.0 <= smoothing <= 1.0:
        raise ValueError("smoothing must be finite and in [0, 1]")
    if not math.isfinite(square_scale) or square_scale < 0.0:
        raise ValueError("lse_square_scale must be finite and nonnegative")
    if not isinstance(logit_scale, float) or not math.isfinite(logit_scale):
        raise ValueError("logit_scale must be a finite Python float")
    if type(ignored) is not int or not -(1 << 63) <= ignored < (1 << 63):
        raise ValueError("ignored_index must be a signed int64 integer")
    if type(inplace) is not bool:
        raise TypeError("inplace_backward must be a bool")
    if process_group is not None:
        raise ValueError("the checked entry requires process_group=None")
    rows, cols = logits.shape
    if cols == 0:
        raise ValueError("cross entropy requires a positive class count")
    block = min(triton.next_power_of_2(cols), 4096)
    if rows > (1 << 31) - 1 or triton.cdiv(cols, block) > 65535:
        raise ValueError("the CUDA grid exceeds its native bounds")
    if rows and (logits.data_ptr() % 4 != 0 or labels.data_ptr() % 8 != 0):
        raise ValueError("logits and labels must have naturally aligned pointers")
    if not bool(((labels == ignored) | ((labels >= 0) & (labels < cols))).all().item()):
        raise ValueError("each label must be ignored_index or a valid class index")


def _check_backward(grad_losses, logits):
    if not isinstance(grad_losses, torch.Tensor):
        raise TypeError("grad_losses must be a Tensor")
    if grad_losses.layout != torch.strided or grad_losses.dtype != torch.float32:
        raise TypeError("grad_losses must be a strided FP32 Tensor")
    if not grad_losses.is_cuda or grad_losses.device != logits.device:
        raise ValueError("grad_losses must use the logits CUDA device")
    if grad_losses.shape != (logits.shape[0],):
        raise ValueError("grad_losses must have shape [rows]")
    if grad_losses.is_neg():
        raise ValueError("grad_losses must have no unresolved negative view bit")
    if grad_losses.numel() and grad_losses.data_ptr() % 4 != 0:
        raise ValueError("grad_losses must have a naturally aligned pointer")


class CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, logit_scale=1.0,
                lse_square_scale=0.0, ignored_index=-100, inplace_backward=False,
                process_group=None):
        _check_forward(logits, labels, smoothing, logit_scale, lse_square_scale,
                       ignored_index, inplace_backward, process_group)
        return _source.CrossEntropyLoss.forward(ctx, logits, labels, smoothing,
            logit_scale, lse_square_scale, ignored_index, inplace_backward, process_group)

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        logits, _, _, _ = ctx.saved_tensors
        _check_backward(grad_losses, logits)
        return _source.CrossEntropyLoss.backward(ctx, grad_losses, grad_z_losses)


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor,
                       label_smoothing: float = 0.0, logit_scale: float = 1.0,
                       lse_square_scale: float = 0.0, ignored_index=-100,
                       inplace_backward: bool = False, process_group=None):
    return CrossEntropyLoss.apply(logits, labels, label_smoothing, logit_scale,
        lse_square_scale, ignored_index, inplace_backward, process_group)

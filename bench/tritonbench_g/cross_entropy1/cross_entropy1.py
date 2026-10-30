"""Logical-view and whole-call repair; original floating expression order retained."""

import math
import torch
import triton
import triton.language as tl
from typing import Tuple

@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0,
    }
)
@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,  # data ptrs
    lse_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,
    lse_square_scale,
    ignored_index,
    total_classes: tl.int64,
    class_start_idx: tl.int64,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols: tl.int64,  # shapes
    n_rows: tl.int64,
    logits_row_stride: tl.int64,  # strides
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    # if SPLIT (e.g. tensor parallel), don't include the LSE in the loss since it's not the final LSE
    SPLIT: tl.constexpr,
):
    # Triton kernel implementation for the forward pass of cross-entropy with label smoothing.
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
    else:
        label_idx -= class_start_idx
        if label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(n_cols, (col_block_idx + 1) * BLOCK_SIZE):
            logits_label = tl.load(logits_ptr + label_idx)
            if HAS_SMOOTHING:
                loss = (
                    (lse if not SPLIT else 0.0)
                    - smoothing * sum_logits / total_classes
                    - (1 - smoothing) * logits_label
                )
            else:
                loss = (lse if not SPLIT else 0.0) - logits_label
        else:
            if HAS_SMOOTHING:
                loss = smoothing * ((lse if not SPLIT else 0.0) - sum_logits / total_classes)
            else:
                loss = 0.0
        if not SPLIT:
            loss += lse_square_scale * lse * lse
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)

@triton.heuristics(
    {
        "HAS_SMOOTHING": lambda args: args["smoothing"] > 0.0,
    }
)
@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    lse_square_scale,
    ignored_index,
    total_classes: tl.int64,
    class_start_idx: tl.int64,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols: tl.int64,  # shapes
    logits_row_stride: tl.int64,  # strides
    dlogits_row_stride: tl.int64,
    dloss_row_stride: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    # Triton kernel implementation for the backward pass of cross-entropy with label smoothing.
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx)
    is_ignored = label_idx == ignored_index
    if not is_ignored:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=-float("inf")).to(tl.float32)
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(is_ignored, class_start_idx, label_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(col_offsets == label_idx, probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(col_offsets == label_idx, probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, dloss * probs), mask=col_offsets < n_cols)

def _check_forward(logits, labels, smoothing, square_scale, ignored, inplace, process_group):
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
    if type(ignored) is not int or not -(1 << 63) <= ignored < (1 << 63):
        raise ValueError("ignored_index must be a signed int64 integer")
    if type(inplace) is not bool:
        raise TypeError("inplace_backward must be a bool")
    if process_group is not None:
        raise ValueError("the checked entry requires process_group=None")
    rows, cols = logits.shape
    if cols == 0:
        raise ZeroDivisionError("integer division or modulo by zero")
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
    def forward(
        ctx,
        logits,
        labels,
        smoothing,
        lse_square_scale=0.0,
        ignored_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        _check_forward(logits, labels, smoothing, lse_square_scale, ignored_index, inplace_backward, process_group)
        # CrossEntropyLoss forward function leveraging the Triton kernel.
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        class_start_idx = rank * n_cols

        labels = labels.resolve_neg().clone(memory_format=torch.contiguous_format)
        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        inplace_logits = logits
        if inplace_backward and n_rows > 1 and logits.stride(0) < n_cols:
            raise ValueError("inplace backward requires non-overlapping logit rows")
        logits = logits.resolve_neg()
        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        MAX_BLOCK_SIZE = 64 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), MAX_BLOCK_SIZE)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else (16 if BLOCK_SIZE < 128 * 1024 else 32))
        split = world_size > 1 or n_cols > MAX_BLOCK_SIZE
        n_splits = (n_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
        loss_shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
        losses = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        lse = torch.empty(*loss_shape, dtype=torch.float, device=logits.device)
        with torch.cuda.device(logits.device.index):
            cross_entropy_fwd_kernel[(n_rows, n_splits)](
                losses,  # data ptrs
                lse,
                logits,
                labels,
                smoothing,
                lse_square_scale,
                ignored_index,
                total_classes,
                class_start_idx,
                n_cols,  # shapes
                n_rows,
                logits.stride(0),  # strides
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
                SPLIT=split,
            )

        if split:
            if n_splits > 1:
                lse = torch.logsumexp(lse, dim=0)
                losses = losses.sum(dim=0)
            if world_size > 1:
                lse_allgather = torch.empty(world_size, n_rows, dtype=lse.dtype, device=lse.device)
                torch.distributed.all_gather_into_tensor(lse_allgather, lse, group=process_group)
                handle_losses = torch.distributed.all_reduce(
                    losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True
                )
                lse = torch.logsumexp(lse_allgather, dim=0)
                handle_losses.wait()
            losses += lse
            if lse_square_scale != 0.0:
                losses += lse_square_scale * lse.square()
            losses.masked_fill_(labels == ignored_index, 0.0)

        ctx.save_for_backward(logits, lse, labels, inplace_logits)
        ctx.smoothing = smoothing
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_losses):
        logits, lse, labels, inplace_logits = ctx.saved_tensors
        _check_backward(grad_losses, logits)
        grad_losses = grad_losses.resolve_neg().clone(memory_format=torch.contiguous_format)
        dlogits = logits if ctx.inplace_backward else torch.empty_like(logits)
        output = inplace_logits if ctx.inplace_backward else dlogits
        if dlogits.stride(-1) != 1:
            dlogits = torch.empty_like(logits, memory_format=torch.contiguous_format)
        n_rows, n_cols = logits.shape
        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        grid = lambda META: (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))  # noqa
        with torch.cuda.device(logits.device.index):
            cross_entropy_bwd_kernel[grid](
                dlogits,  # data ptrs
                grad_losses,
                logits,
                lse,
                labels,
                ctx.smoothing,
                ctx.lse_square_scale,
                ctx.ignored_index,
                ctx.total_classes,
                ctx.class_start_idx,
                n_cols,  # shapes
                logits.stride(0),  # strides
                dlogits.stride(0),
                grad_losses.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,  # constants
                num_warps=num_warps,
            )
        if dlogits is not output:
            output.copy_(dlogits)
        return output, None, None, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    lse_square_scale: float = 0.0,
    ignored_index=-100,
    inplace_backward: bool = False,
    process_group=None,
) -> torch.Tensor:
    """
    CrossEntropyLoss wrapper function for calling the custom autograd Function.
    """
    return CrossEntropyLoss.apply(
        logits,
        labels,
        label_smoothing,
        lse_square_scale,
        ignored_index,
        inplace_backward,
        process_group,
    )

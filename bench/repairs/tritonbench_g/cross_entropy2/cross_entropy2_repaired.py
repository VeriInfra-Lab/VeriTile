"""Complete row outputs for the explicit scaled cross-entropy stages."""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def cross_entropy_fwd_kernel(
    loss_ptr,  # data ptrs
    lse_ptr,
    z_loss_ptr,
    logits_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index: tl.int64,
    total_classes: tl.int64,
    class_start_idx: tl.int64,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols: tl.int64,  # shapes
    n_rows: tl.int64,
    logits_row_stride: tl.int64,  # strides
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
    SPLIT: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
    owns_target = (label_idx >= class_start_idx) & (label_idx < class_start_idx + n_cols)
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    max_logits = tl.max(logits, 0)
    if HAS_SMOOTHING:
        sum_logits = tl.sum(tl.where(col_offsets < n_cols, logits, 0.0), 0)
    lse = tl.log(tl.sum(tl.exp(logits - max_logits), 0)) + max_logits
    tl.store(lse_ptr + col_block_idx * n_rows + row_idx, lse)
    if label_idx == ignored_index:
        loss = 0.0
        z_loss = 0.0
    else:
        label_idx = tl.where(owns_target, label_idx, class_start_idx) - class_start_idx
        if owns_target and label_idx >= col_block_idx * BLOCK_SIZE and label_idx < min(
            n_cols, (col_block_idx + 1) * BLOCK_SIZE
        ):
            logits_label = tl.load(logits_ptr + label_idx) * logit_scale
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
                if not SPLIT:
                    loss += (1 - smoothing) * lse
            else:
                loss = lse if not SPLIT else 0.0
        if not SPLIT:
            z_loss = lse_square_scale * lse * lse
            loss += z_loss
        else:
            z_loss = 0.0
    tl.store(loss_ptr + col_block_idx * n_rows + row_idx, loss)
    tl.store(z_loss_ptr + col_block_idx * n_rows + row_idx, z_loss)

@triton.jit
def cross_entropy_bwd_kernel(
    dlogits_ptr,  # data ptrs
    dloss_ptr,
    logits_ptr,
    lse_ptr,
    labels_ptr,
    smoothing,
    logit_scale,
    lse_square_scale,
    ignored_index: tl.int64,
    total_classes: tl.int64,
    class_start_idx: tl.int64,  # Useful for tensor parallel when each rank only has a subset of classes
    n_cols: tl.int64,  # shapes
    logits_row_stride: tl.int64,  # strides
    dlogits_row_stride: tl.int64,
    dloss_row_stride: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    HAS_SMOOTHING: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_block_idx = tl.program_id(1).to(tl.int64)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)
    owns_target = (label_idx >= class_start_idx) & (label_idx < class_start_idx + n_cols)
    is_ignored = label_idx == ignored_index
    if not is_ignored:
        dloss = tl.load(dloss_ptr + row_idx * dloss_row_stride)
    else:
        dloss = 0.0
    logits = tl.load(logits_ptr + col_offsets, mask=col_offsets < n_cols, other=0.0).to(
        tl.float32
    ) * logit_scale
    logits = tl.where(col_offsets < n_cols, logits, -float("inf"))
    lse = tl.load(lse_ptr + row_idx)
    probs = tl.exp(logits - lse)
    probs += 2.0 * lse_square_scale * lse * probs
    label_idx = tl.where(owns_target, label_idx, class_start_idx) - class_start_idx
    if HAS_SMOOTHING:
        smooth_negative = smoothing / total_classes
        probs = tl.where(owns_target & (col_offsets == label_idx), probs - (1 - smoothing), probs) - smooth_negative
    else:
        probs = tl.where(owns_target & (col_offsets == label_idx), probs - 1.0, probs)
    tl.store(dlogits_ptr + col_offsets, tl.where(is_ignored, 0.0, (dloss * logit_scale) * probs), mask=col_offsets < n_cols)


def _validate(logits, labels, smoothing, logit_scale, lse_square_scale,
              ignored_index, total_classes, class_start_idx, BLOCK_SIZE,
              HAS_SMOOTHING):
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("logits must have shape (rows, positive classes)")
    if logits.dtype != torch.float32 or labels.dtype != torch.int32:
        raise TypeError("selected source precision requires float32 logits and int32 labels")
    if labels.shape != (logits.shape[0],) or labels.device != logits.device:
        raise ValueError("labels must have one entry per row on the logits device")
    if type(BLOCK_SIZE) is not int or BLOCK_SIZE <= 0 or BLOCK_SIZE > 65536 or BLOCK_SIZE & (BLOCK_SIZE - 1):
        raise ValueError("BLOCK_SIZE must be a supported positive power of two")
    if type(HAS_SMOOTHING) is not bool:
        raise TypeError("HAS_SMOOTHING must be bool")
    if any(type(x) is not int for x in (ignored_index, total_classes, class_start_idx)):
        raise TypeError("class indices must be integers")
    if not (-2**63 <= ignored_index < 2**63 and 0 < total_classes < 2**63
            and 0 <= class_start_idx and class_start_idx + logits.shape[1] <= total_classes):
        raise ValueError("class metadata is outside the signed int64 class interval")
    if not (math.isfinite(smoothing) and 0 <= smoothing <= 1
            and math.isfinite(logit_scale) and math.isfinite(lse_square_scale)
            and lse_square_scale >= 0):
        raise ValueError("invalid smoothing, logit scale, or LSE-square scale")


def _checked_validate(logits, labels, smoothing, logit_scale, lse_square_scale,
                      ignored_index, total_classes, class_start_idx, BLOCK_SIZE,
                      HAS_SMOOTHING):
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be Tensors")
    for name, value in (("logits", logits), ("labels", labels)):
        if value.layout != torch.strided or not value.is_cuda:
            raise ValueError(name + " must be a strided CUDA Tensor")
        if not value.is_contiguous() or value.is_neg():
            raise ValueError(name + " must be contiguous with no negative view bit")
        if value.numel() and value.data_ptr() % 4:
            raise ValueError(name + " must have a naturally aligned pointer")
    if not all(isinstance(v, float) for v in (smoothing, logit_scale, lse_square_scale)):
        raise TypeError("smoothing, logit_scale, and lse_square_scale must be Python floats")
    _validate(logits, labels, smoothing, logit_scale, lse_square_scale,
              ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING)
    if logits.device.index != torch.cuda.current_device():
        raise ValueError("logits must be on the current CUDA launch device")
    rows, cols = logits.shape
    if rows and (rows > (1 << 31) - 1 or triton.cdiv(cols, BLOCK_SIZE) > 65535):
        raise ValueError("the CUDA grid exceeds its native bounds")
    labels64 = labels.to(dtype=torch.int64)
    if not bool(((labels64 == ignored_index) | ((labels64 >= 0) & (labels64 < total_classes))).all().item()):
        raise ValueError("each label must be ignored_index or a valid global class index")


def _check_row(value, logits):
    if not isinstance(value, torch.Tensor):
        raise TypeError("upstream and LSE must be Tensors")
    if value.layout != torch.strided or not value.is_cuda:
        raise ValueError("upstream and LSE must be strided CUDA Tensors")
    if not value.is_contiguous() or value.is_neg():
        raise ValueError("upstream and LSE must be contiguous with no negative view bit")
    if value.numel() and value.data_ptr() % 4:
        raise ValueError("upstream and LSE must have naturally aligned pointers")


def cross_entropy_fwd(
    logits, labels, smoothing, logit_scale, lse_square_scale, ignored_index,
    total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING, SPLIT
):
    _checked_validate(logits, labels, smoothing, logit_scale, lse_square_scale,
              ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING)
    if type(SPLIT) is not bool:
        raise TypeError("SPLIT must be bool")
    n_rows, n_cols = logits.shape
    logits = logits.resolve_neg().contiguous()
    labels = labels.resolve_neg().contiguous()
    n_splits = triton.cdiv(n_cols, BLOCK_SIZE)
    # SPLIT defers normalization to the host. Multiple blocks or an incomplete
    # class partition also require that path, regardless of the requested flag.
    split = SPLIT or n_splits > 1 or class_start_idx != 0 or n_cols != total_classes
    shape = (n_splits, n_rows) if n_splits > 1 else (n_rows,)
    loss = torch.empty(shape, dtype=torch.float32, device=logits.device)
    lse = torch.empty(shape, dtype=torch.float32, device=logits.device)
    z_loss = torch.empty(shape, dtype=torch.float32, device=logits.device)
    if n_rows:
        cross_entropy_fwd_kernel[(n_rows, n_splits)](
            loss, lse, z_loss, logits, labels, smoothing, logit_scale,
            lse_square_scale, ignored_index, total_classes, class_start_idx,
            n_cols, n_rows, logits.stride(0), BLOCK_SIZE, HAS_SMOOTHING, split
        )
    if split:
        if n_splits > 1:
            lse = torch.logsumexp(lse, dim=0)
            loss = loss.sum(dim=0)
        loss += lse
        z_loss = (lse_square_scale * lse) * lse
        loss += z_loss
        ignored = labels.to(dtype=torch.int64) == ignored_index
        loss.masked_fill_(ignored, 0.0)
        z_loss.masked_fill_(ignored, 0.0)
    return loss, lse, z_loss


def cross_entropy_bwd(
    dloss, logits, lse, labels, smoothing, logit_scale, lse_square_scale,
    ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING
):
    _checked_validate(logits, labels, smoothing, logit_scale, lse_square_scale,
              ignored_index, total_classes, class_start_idx, BLOCK_SIZE, HAS_SMOOTHING)
    n_rows, n_cols = logits.shape
    for name, value in (("dloss", dloss), ("lse", lse)):
        _check_row(value, logits)
        if value.dtype != torch.float32 or value.shape != (n_rows,) or value.device != logits.device:
            raise ValueError(name + " must be a float32 row vector on the logits device")
    logits = logits.resolve_neg().contiguous()
    labels = labels.resolve_neg().contiguous()
    dloss = dloss.resolve_neg().contiguous()
    lse = lse.resolve_neg().contiguous()
    dlogits = torch.empty_like(logits, memory_format=torch.contiguous_format)
    if n_rows:
        cross_entropy_bwd_kernel[(n_rows, triton.cdiv(n_cols, BLOCK_SIZE))](
            dlogits, dloss, logits, lse, labels, smoothing, logit_scale,
            lse_square_scale, ignored_index, total_classes, class_start_idx,
            n_cols, logits.stride(0), dlogits.stride(0), dloss.stride(0),
            BLOCK_SIZE, HAS_SMOOTHING
        )
    return dlogits

"""Checked FP32/int64 entry for the explicitly ignored-zero FastCE body."""
import math
import torch



def _check_scalar(value, name):
    if type(value) is int:
        if not -(1 << 63) <= value < (1 << 63):
            raise ValueError(name + " must fit signed int64")
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError(name + " must be finite")
    else:
        raise TypeError(name + " must be a Python int or float")


def _storage_id(tensor):
    return tensor.untyped_storage()._cdata


def _check_forward(logits, labels, logit_softcapping, logit_scaling, n_items):
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be Tensors")
    if logits.layout != torch.strided or labels.layout != torch.strided:
        raise TypeError("logits and labels must have strided layout")
    if logits.dtype != torch.float32 or labels.dtype != torch.int64:
        raise TypeError("expected FP32 logits and int64 labels")
    if not logits.is_cuda or not labels.is_cuda or logits.device != labels.device:
        raise ValueError("logits and labels must share one CUDA device")
    if logits.device.index != torch.cuda.current_device():
        raise ValueError("the logits device must be the current CUDA device")
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("expected logits [batch, sequence, vocabulary] and matching labels")
    if not logits.is_contiguous() or not labels.is_contiguous():
        raise ValueError("logits and labels must be contiguous")
    if logits.is_neg() or logits.is_conj() or labels.is_neg() or labels.is_conj():
        raise ValueError("inputs must have resolved negative and conjugate bits")
    if _storage_id(logits) == _storage_id(labels):
        raise ValueError("logits and labels must use distinct storage allocations")
    batch, sequence, vocab = logits.shape
    rows = batch * sequence
    if vocab == 0:
        raise ValueError("vocabulary must be positive")
    if rows > (1 << 31) - 1 or (vocab + 4095) // 4096 > 65535:
        raise ValueError("the CUDA grid exceeds its native bounds")
    if rows and (logits.data_ptr() % 4 or labels.data_ptr() % 8):
        raise ValueError("logits and labels must have naturally aligned pointers")
    _check_scalar(logit_softcapping, "logit_softcapping")
    _check_scalar(logit_scaling, "logit_scaling")
    if not bool(((labels == -100) | ((labels >= 0) & (labels < vocab))).all().item()):
        raise ValueError("each label must be -100 or a valid class index")
    if n_items is None:
        if torch.count_nonzero(labels != -100).item() == 0:
            raise ValueError("the default denominator requires a nonignored label")
    else:
        _check_scalar(n_items, "n_items")
        if n_items <= 0:
            raise ValueError("n_items must be positive")


def _check_upstream(upstream, logits, labels):
    if not isinstance(upstream, torch.Tensor):
        raise TypeError("upstream must be a Tensor")
    if upstream.layout != torch.strided or upstream.dtype != torch.float32:
        raise TypeError("upstream must be strided FP32")
    if not upstream.is_cuda or upstream.device != logits.device:
        raise ValueError("upstream must use the logits CUDA device")
    if logits.device.index != torch.cuda.current_device():
        raise ValueError("the logits device must be the current CUDA device")
    if upstream.ndim != 0 or upstream.is_neg() or upstream.is_conj():
        raise ValueError("upstream must be a resolved scalar")
    if upstream.data_ptr() % 4:
        raise ValueError("upstream must have a naturally aligned pointer")
    if _storage_id(upstream) == _storage_id(labels):
        raise ValueError("upstream and labels must use distinct storage allocations")
    return upstream


def fast_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0, n_items=None):
    _check_forward(logits, labels, logit_softcapping, logit_scaling, n_items)
    scalar = _unchecked_fast_cross_entropy_loss(logits, labels, logit_softcapping, logit_scaling, n_items)
    if scalar.requires_grad:
        scalar.register_hook(lambda upstream: _check_upstream(upstream, logits, labels))
    return scalar


import triton
import triton.language as tl
from triton.language.extra import libdevice
triton_tanh = libdevice.tanh

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_forward_index_commuted(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_ptr += logits_row_stride.to(tl.int64) * row_idx
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _chunked_cross_entropy_forward_index_commuted(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1).to(tl.int64)
    logits_ptr += logits_row_stride.to(tl.int64) * row_idx
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
            if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_backward_index_commuted(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1).to(tl.int64)

    logits_ptr += logits_row_stride.to(tl.int64) * row_idx
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    if DO_SOFTCAPPING:
        y = y * (1.0 - partial*partial)

    tl.store(logits_ptr + col_offsets, tl.where(label_idx == -100, 0.0, dloss * y), mask=mask)



# from .utils import triton_tanh

next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE : int = 65536

def calculate_settings(n : int) -> (int, int):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx
    labels_ptr += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)

@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    chunk_idx = tl.program_id(1).to(tl.int64)
    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int64)
    logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = tl.where(mask, logits.to(tl.float32), -float("inf"))
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
            if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    tl.store(logsumexp_ptr, logsumexp)


@triton.heuristics({
    "DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],
    "DO_LOGIT_SCALING": lambda args: args["DO_LOGIT_SCALING"],
})
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr, dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_SOFTCAPPING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    DO_LOGIT_SCALING: tl.constexpr,
    LOGIT_SCALE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    block_idx = tl.program_id(1).to(tl.int64)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr += row_idx * dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int64)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf"))

    if DO_LOGIT_SCALING:
        x = x * LOGIT_SCALE

    if DO_SOFTCAPPING:
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0,
        y,
    )

    if DO_LOGIT_SCALING:
        y = y * LOGIT_SCALE

    if DO_SOFTCAPPING:
        y = y * (1.0 - partial*partial)

    tl.store(logits_ptr + col_offsets, tl.where(label_idx == -100, 0.0, dloss * y), mask=mask)

MAX_FUSED_SIZE = 65536

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping=0, logit_scaling=0):
        n_rows, vocab_size = logits.shape

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        DO_SOFTCAPPING = (logit_softcapping != 0)
        DO_LOGIT_SCALING = (logit_scaling != 0)

        if n_chunks == 1:
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=DO_LOGIT_SCALING,
                LOGIT_SCALE=logit_scaling,
                num_warps=num_warps,
            )
        else:
            logsumexp = torch.empty((n_rows, n_chunks,), dtype=torch.float32, device=logits.device)

            _chunked_cross_entropy_forward[(n_rows, n_chunks,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE=vocab_size,
                N_CHUNKS=n_chunks,
                BLOCK_SIZE=MAX_FUSED_SIZE,
                DO_SOFTCAPPING=DO_SOFTCAPPING,
                SOFTCAP=logit_softcapping,
                DO_LOGIT_SCALING=DO_LOGIT_SCALING,
                LOGIT_SCALE=logit_scaling,
                num_warps=32,
            )
            logsumexp = torch.logsumexp(logsumexp, dim=1)
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0)

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        ctx.DO_LOGIT_SCALING = DO_LOGIT_SCALING
        ctx.logit_scaling = logit_scaling
        return losses

    @staticmethod
    def backward(ctx, dlosses):
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        _cross_entropy_backward[(n_rows, n_blocks,)](
            logits, logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE=vocab_size,
            BLOCK_SIZE=BLOCK_SIZE,
            DO_SOFTCAPPING=ctx.DO_SOFTCAPPING,
            SOFTCAP=ctx.logit_softcapping,
            DO_LOGIT_SCALING=ctx.DO_LOGIT_SCALING,
            LOGIT_SCALE=ctx.logit_scaling,
            num_warps=8,
        )
        return logits, None, None, None,

def _unchecked_fast_cross_entropy_loss(
    logits,
    labels,
    logit_softcapping=0,
    logit_scaling=0,
    n_items=None,
):
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch*seq_len, d),
        labels.view(-1),
        logit_softcapping,
        logit_scaling,
    )
    if n_items is None:
        n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items

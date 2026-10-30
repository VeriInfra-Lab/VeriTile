"""Checked FP32/int64 entry for the explicitly ignored-zero FastCE body."""
import importlib.util
import math
from pathlib import Path
import torch

_source_spec = importlib.util.spec_from_file_location(
    "_veritile_fast_ce_source", Path(__file__).with_name("fast_ce_loss_body.py"))
_source = importlib.util.module_from_spec(_source_spec)
_source_spec.loader.exec_module(_source)


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
    scalar = _source.fast_cross_entropy_loss(logits, labels, logit_softcapping, logit_scaling, n_items)
    if scalar.requires_grad:
        scalar.register_hook(lambda upstream: _check_upstream(upstream, logits, labels))
    return scalar

"""FP32 three-pass layer norm with actual inner strides and int64 indices."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_kernel(
    X, W, Y, stride_x_N, stride_x_hn, stride_x_hd,
    stride_y_N, stride_y_hn, stride_y_hd, stride_w_hn, stride_w_hd,
    N, eps, BLOCK_SIZE: tl.constexpr,
):
    Seq = tl.program_id(0).to(tl.int64)
    H = tl.program_id(1).to(tl.int64)
    stride_x_N = tl.cast(stride_x_N, tl.int64)
    stride_x_hn = tl.cast(stride_x_hn, tl.int64)
    stride_x_hd = tl.cast(stride_x_hd, tl.int64)
    stride_y_N = tl.cast(stride_y_N, tl.int64)
    stride_y_hn = tl.cast(stride_y_hn, tl.int64)
    stride_y_hd = tl.cast(stride_y_hd, tl.int64)
    stride_w_hn = tl.cast(stride_w_hn, tl.int64)
    stride_w_hd = tl.cast(stride_w_hd, tl.int64)
    N = tl.cast(N, tl.int64)
    zero = tl.full((), 0, tl.int64)
    step = tl.full((), BLOCK_SIZE, tl.int64)
    lanes = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    X += Seq * stride_x_N + H * stride_x_hn
    Y += Seq * stride_y_N + H * stride_y_hn
    W += H * stride_w_hn

    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(zero, N, step):
        cols = off + lanes
        a = tl.load(X + cols * stride_x_hd, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(zero, N, step):
        cols = off + lanes
        x = tl.load(X + cols * stride_x_hd, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    for off in range(zero, N, step):
        cols = off + lanes
        mask = cols < N
        w = tl.load(W + cols * stride_w_hd, mask=mask).to(tl.float32)
        x = tl.load(X + cols * stride_x_hd, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w
        tl.store(Y + cols * stride_y_hd, y.to(X.dtype.element_ty), mask=mask)


def _checked_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a Tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError("expected a strided FP32 Tensor")
    if tensor.device.type != "cuda":
        raise ValueError("expected a CUDA Tensor")
    if tensor.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if tensor.numel() and tensor.data_ptr() % 4:
        raise ValueError("FP32 data pointer must be four-byte aligned")


def layernorm_forward(X, W, eps):
    _checked_tensor(X)
    _checked_tensor(W)
    if X.ndim != 3 or W.ndim != 2 or W.shape != X.shape[1:]:
        raise ValueError("expected X[S,H,N] and W[H,N]")
    if W.device != X.device:
        raise ValueError("X and W must be on the same device")
    try:
        valid_eps = isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0
    except (OverflowError, TypeError, ValueError):
        valid_eps = False
    if not valid_eps:
        raise ValueError("epsilon must be a finite positive Python scalar")
    with torch.cuda.device(X.device):
        y = torch.empty_like(X)
        _layer_norm_fwd_kernel[(X.shape[0], X.shape[1])](
            X, W, y, X.stride(0), X.stride(1), X.stride(2),
            y.stride(0), y.stride(1), y.stride(2), W.stride(0), W.stride(1),
            X.shape[-1], eps, 128,
        )
    return y

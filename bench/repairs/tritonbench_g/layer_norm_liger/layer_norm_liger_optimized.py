"""Related source candidates proved by FPEquiv.lean.

Forward shares the repeated centered vector. Backward hoists the repeated FP32
column conversion. No measured performance improvement is claimed.
"""

import math
import torch
import triton
import triton.language as tl
import functools

def is_hip() -> bool:
    return torch.version.hip is not None

def ensure_contiguous(fn):
    @functools.wraps(fn)
    def wrapper(ctx, *args, **kwargs):
        def maybe_to_contiguous(x):
            if isinstance(x, torch.Tensor):
                _checked_tensor(x)
                return x.contiguous()
            return x

        args = [maybe_to_contiguous(arg) for arg in args]
        kwargs = {k: maybe_to_contiguous(v) for k, v in kwargs.items()}
        return fn(ctx, *args, **kwargs)

    return wrapper


def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    Y_row_stride = tl.cast(Y_row_stride, tl.int64)
    X_row_stride = tl.cast(X_row_stride, tl.int64)
    Mean_row_stride = tl.cast(Mean_row_stride, tl.int64)
    RSTD_row_stride = tl.cast(RSTD_row_stride, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    centered = X_row - mean
    var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / n_cols
    rstd = tl.rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    Y_row = centered * rstd * W_row + B_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)

@triton.jit
def _layer_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    stride_dx,  # stride of each row in input grad
    stride_dw,  # stride of each row in weights grad
    stride_db,  # stride of each row in bias grad
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    stride_x = tl.cast(stride_x, tl.int64)
    stride_dx = tl.cast(stride_dx, tl.int64)
    stride_dw = tl.cast(stride_dw, tl.int64)
    stride_db = tl.cast(stride_db, tl.int64)
    stride_dy = tl.cast(stride_dy, tl.int64)
    n_rows = tl.cast(n_rows, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    row_block_id = tl.program_id(0).to(tl.int64)
    group = tl.full((), rows_per_program, tl.int64)
    one = tl.full((), 1, tl.int64)
    row_start = row_block_id * group
    row_end = min((row_block_id + one) * group, n_rows)
    cols = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    denominator = n_cols.to(tl.float32)
    for _ in range(row_start, row_end, one):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / denominator
        c2 = tl.sum(wdy, axis=0) / denominator
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)

def _checked_tensor(t):
    if not isinstance(t, torch.Tensor):
        raise TypeError("expected a Tensor")
    if t.layout != torch.strided or t.dtype != torch.float32:
        raise TypeError("only strided FP32 tensors are supported")
    if not t.is_cuda:
        raise ValueError("expected a CUDA tensor")
    if t.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if t.numel() and t.data_ptr() % 4:
        raise ValueError("nonempty FP32 tensors require 4-byte aligned pointers")


def _checked_inputs(X, W, B, eps):
    """Selected public contract: nonempty FP32 input and matching FP32 vectors."""
    for t in (X, W, B):
        _checked_tensor(t)
    if X.ndim < 1 or X.numel() == 0:
        raise ValueError("X must have positive dimensions and a nonempty last axis")
    n = X.shape[-1]
    if n <= 0 or n > 65536 or W.ndim != 1 or B.ndim != 1 or W.numel() != n or B.numel() != n:
        raise ValueError("W and B must be vectors matching the last dimension, 1 <= N <= 65536")
    if any(t.dtype != torch.float32 for t in (X, W, B)):
        raise TypeError("this selected implementation requires FP32 X, W and B")
    if W.device != X.device or B.device != X.device:
        raise ValueError("X, W and B must be on the same device")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("epsilon must be a finite positive scalar")
    return X.contiguous(), W.contiguous(), B.contiguous()

def layer_norm_forward(X, W, B, eps):
    X, W, B = _checked_inputs(X, W, B, eps)
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    assert (
        X.shape[1] == W.shape[0]
    ), f"Incompatible hidden size dimension between input tensor with shape[1] = {X.shape[1]} and weight tensor with shape[0] = {W.shape[0]}"

    with torch.cuda.device(X.device):
        _layer_norm_forward_kernel[(n_rows,)](
            Y,
            Y.stride(0),
            X,
            X.stride(0),
            W,
            W.stride(0),
            B,
            B.stride(0),
            Mean,
            Mean.stride(0),
            RSTD,
            RSTD.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps

def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    # X/W/B/Mean/RSTD are the saved tensors from this forward call.
    _checked_tensor(dY)
    if dY.ndim < 1 or dY.dtype != torch.float32 or dY.device != X.device or dY.numel() != X.numel() or dY.shape[-1] != X.shape[-1]:
        raise ValueError("dY must match the saved FP32 input shape and device")
    dY = dY.contiguous()
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    if sm_count <= 0:
        raise RuntimeError("device must provide a positive program count")
    _DW = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)
    _DB = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    rows_per_program = (n_rows + sm_count - 1) // sm_count
    grid = (sm_count,)
    triton_dtype = tl.float32 if X.dtype == torch.float32 else tl.bfloat16
    with torch.cuda.device(X.device):
        _layer_norm_backward_kernel[grid](
            X,
            W,
            Mean,
            RSTD,
            DX,
            _DW,
            _DB,
            dY,
            X.stride(0),
            DX.stride(0),
            _DW.stride(0),
            _DB.stride(0),
            dY.stride(0),
            n_rows,
            n_cols,
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
            dtype=triton_dtype,
        )

    DW = _DW.sum(dim=0).to(W.dtype)
    DB = _DB.sum(dim=0).to(W.dtype)

    DX = DX.view(*shape)
    return DX, DW, DB

class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None

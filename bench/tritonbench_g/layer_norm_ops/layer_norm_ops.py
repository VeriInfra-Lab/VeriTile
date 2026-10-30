
import torch
import triton
import triton.language as tl
import math



def _check_tensor(tensor, shape, strides, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise ValueError(f"{name} must be strided FP32")
    if not tensor.is_cuda or tensor.device.index != 0:
        raise ValueError(f"{name} must be on CUDA device 0")
    if tensor.shape != shape or tensor.stride() != strides:
        raise ValueError(f"{name} must have the declared shape and canonical strides")
    if tensor.storage_offset() != 0 or tensor.is_neg():
        raise ValueError(f"{name} must have zero storage offset and no negative view flag")
    if tensor.numel() > 0 and tensor.data_ptr() % 4 != 0:
        raise ValueError(f"{name} requires 4-byte aligned nonempty storage")


def _check_common(x, weight, bias, eps, is_rms_norm):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.ndim != 2:
        raise ValueError("x must have shape (M, N)")
    M, N = x.shape
    if not 1 <= N <= 16384:
        raise ValueError("N must lie in [1, 16384]")
    _check_tensor(x, (M, N), (N, 1), "x")
    _check_tensor(weight, (N,), (1,), "weight")
    if bias is not None:
        _check_tensor(bias, (N,), (1,), "bias")
    if torch.cuda.current_device() != 0:
        raise ValueError("the current CUDA device must be 0")
    if type(eps) is not float or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive Python float")
    if type(is_rms_norm) is not bool:
        raise TypeError("is_rms_norm must be a bool")
    block = min(16384, triton.next_power_of_2(N))
    if M * N + block >= 2**31:
        raise ValueError("forward padded source indices must fit signed int32")
    return M, N, block


def _check_forward(x, weight, bias, eps, residual, out_dtype, residual_dtype, is_rms_norm):
    M, N, _ = _check_common(x, weight, bias, eps, is_rms_norm)
    if out_dtype is not None and out_dtype != torch.float32:
        raise ValueError("out_dtype must be None or FP32")
    if residual is not None:
        _check_tensor(residual, (M, N), (N, 1), "residual")
    elif residual_dtype is not None and residual_dtype != torch.float32:
        raise ValueError("residual_dtype must be None or FP32 when no residual overrides it")


def _check_backward(dy, x, weight, bias, eps, mean, rstd, dresidual,
                    has_residual, is_rms_norm, x_dtype, recompute_output):
    M, N, block = _check_common(x, weight, bias, eps, is_rms_norm)
    _check_tensor(dy, (M, N), (N, 1), "dy")
    if dresidual is not None:
        _check_tensor(dresidual, (M, N), (N, 1), "dresidual")
    if is_rms_norm:
        if mean is not None:
            raise ValueError("RMS mode uses no Mean tensor")
    else:
        _check_tensor(mean, (M,), (1,), "mean")
    _check_tensor(rstd, (M,), (1,), "rstd")
    if x_dtype is not None and x_dtype != torch.float32:
        raise ValueError("x_dtype must be None or FP32")
    if type(has_residual) is not bool or type(recompute_output) is not bool:
        raise TypeError("has_residual and recompute_output must be bools")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    if sm_count <= 0:
        raise ValueError("the source launch requires a positive SM count")
    rows_per_program = math.ceil(M / sm_count)
    if rows_per_program != (M + sm_count - 1) // sm_count:
        raise ValueError("the source row grouping must equal integer ceiling division")
    if sm_count * rows_per_program * N + block >= 2**31 or sm_count * N + block >= 2**31:
        raise ValueError("backward padded source indices must fit signed int32")
    return sm_count


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.jit
def _layer_norm_fwd_1pass_kernel(
    X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean, Rstd, 
    stride_x_row, stride_y_row, stride_res_row, stride_res_out_row, 
    N, eps, IS_RMS_NORM: tl.constexpr, BLOCK_N: tl.constexpr, 
    HAS_RESIDUAL: tl.constexpr, STORE_RESIDUAL_OUT: tl.constexpr, HAS_BIAS: tl.constexpr
):
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    mask = cols < N
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
    y = x_hat * w + b if HAS_BIAS else x_hat * w
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd(
    x, weight, bias, eps, residual=None, out_dtype=None, 
    residual_dtype=None, is_rms_norm=False
):
    _check_forward(x, weight, bias, eps, residual, out_dtype, residual_dtype, is_rms_norm)
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device="cuda") if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_1pass_kernel[(M,)](
            x, y, weight, bias, residual, residual_out, 
            mean, rstd, x.stride(0), y.stride(0), 
            residual.stride(0) if residual is not None else 0, 
            residual_out.stride(0) if residual_out is not None else 0, 
            N, eps, is_rms_norm, BLOCK_N, residual is not None, 
            residual_out is not None, bias is not None
        )
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X, W, B, Y, DY, DX, DW, DB, DRESIDUAL, DRESIDUAL_IN, 
    Mean, Rstd, stride_x_row, stride_y_row, stride_dy_row, 
    stride_dx_row, stride_dres_row, stride_dres_in_row, stride_mean, stride_rstd, M, 
    N, eps, rows_per_program, IS_RMS_NORM: tl.constexpr, 
    BLOCK_N: tl.constexpr, HAS_DRESIDUAL: tl.constexpr, 
    STORE_DRESIDUAL: tl.constexpr, HAS_BIAS: tl.constexpr, 
    RECOMPUTE_OUTPUT: tl.constexpr
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row * stride_mean) if Mean is not None else 0.0  # Use zero when Mean is absent.
        rstd = tl.load(Rstd + row * stride_rstd)
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w + b if HAS_BIAS else xhat * w
            tl.store(Y + cols, y, mask=mask)
        wdy = w * dy
        dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)
        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def _layer_norm_bwd(
    dy, x, weight, bias, eps, mean, rstd, dresidual=None, 
    has_residual=False, is_rms_norm=False, x_dtype=None, recompute_output=False
):
    sm_count = _check_backward(dy, x, weight, bias, eps, mean, rstd, dresidual,
                               has_residual, is_rms_norm, x_dtype, recompute_output)
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    dx = (
        torch.empty_like(x)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    dresidual_in = torch.empty_like(x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype, device=dy.device) if recompute_output else None
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    _dw = torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
    _db = (
        torch.empty((sm_count, N), dtype=torch.float32, device=bias.device)
        if bias is not None
        else None
    )
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x, weight, bias, y, dy, dx, _dw, _db, dresidual, 
            dresidual_in, mean, rstd, x.stride(0), 
            0 if not recompute_output else y.stride(0), dy.stride(0), 
            dx.stride(0), dresidual.stride(0) if dresidual is not None else 0, 
            dresidual_in.stride(0) if dresidual_in is not None else 0, 
            mean.stride(0) if mean is not None else 0, rstd.stride(0),
            M, N, eps, rows_per_program, is_rms_norm, BLOCK_N, 
            dresidual is not None, dresidual_in is not None, bias is not None
        )
    dw = _dw.sum(0).to(weight.dtype)
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)

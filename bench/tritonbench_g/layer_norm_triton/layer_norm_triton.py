"""Checked FP32 LayerNorm with complete forward and grouped backward repair."""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _layer_norm_fwd_fused(X, Y, W, B, Mean, Rstd, stride, N, eps,
                          BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    stride = tl.cast(stride, tl.int64)
    N = tl.cast(N, tl.int64)
    Y += row * stride
    X += row * stride
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(DX, DY, DW, DB, X, W, Mean, Rstd,
                             stride, M, N, GROUP_SIZE_M: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr):
    # One program owns one original accumulation group. Increasing rows select
    # a concrete legal order without a spin lock or cross-program publication.
    group = tl.program_id(0).to(tl.int64)
    stride = tl.cast(stride, tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    cols = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    mask = cols < N
    DW = DW + group * N + cols
    DB = DB + group * N + cols
    dw_acc = tl.zeros([BLOCK_SIZE_N], dtype=DW.dtype.element_ty)
    db_acc = tl.zeros([BLOCK_SIZE_N], dtype=DB.dtype.element_ty)
    row = group
    while row < M:
        x = tl.load(X + row * stride + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + row * stride + cols, mask=mask, other=0).to(tl.float32)
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        xhat = (x - mean) * rstd
        wdy = w * dy
        xhat = tl.where(mask, xhat, 0.)
        wdy = tl.where(mask, wdy, 0.)
        c1 = tl.sum(xhat * wdy, axis=0) / N
        c2 = tl.sum(wdy, axis=0) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd
        tl.store(DX + row * stride + cols, dx, mask=mask)
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = dy.to(w.dtype)
        if row != group:
            partial_dw += dw_acc
            partial_db += db_acc
        # Retain each original partial-storage rounding in the carried value.
        dw_acc = partial_dw.to(DW.dtype.element_ty)
        db_acc = partial_db.to(DB.dtype.element_ty)
        row += GROUP_SIZE_M
    tl.store(DW, dw_acc, mask=mask)
    tl.store(DB, db_acc, mask=mask)


@triton.jit
def _layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, M, N,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.)
        db += tl.load(DB + offs, mask=mask, other=0.)
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


def _check_tensor(tensor, device=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a Tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError("expected strided FP32 storage")
    if tensor.device.type != "cuda":
        raise ValueError("expected CUDA storage")
    if device is not None and tensor.device != device:
        raise ValueError("all inputs must share a CUDA device")
    if not tensor.is_contiguous():
        raise ValueError("expected contiguous storage")
    if tensor.is_neg():
        raise ValueError("lazy-negative tensors are unsupported")
    if tensor.numel() and tensor.data_ptr() % 4:
        raise ValueError("expected four-byte-aligned storage")


def _check_inputs(x, normalized_shape, weight, bias, eps):
    _check_tensor(x)
    _check_tensor(weight, x.device)
    _check_tensor(bias, x.device)
    if type(eps) is not float or not math.isfinite(eps) or not eps > 0.0:
        raise ValueError("eps must be a finite positive builtin float")
    if type(normalized_shape) is int:
        normalized_shape = (normalized_shape,)
    elif isinstance(normalized_shape, (tuple, list)):
        normalized_shape = tuple(normalized_shape)
    else:
        raise TypeError("normalized_shape must be an integer or integer sequence")
    if any(type(d) is not int or d < 0 for d in normalized_shape):
        raise ValueError("normalized_shape must contain nonnegative builtin integers")
    _shape(x, normalized_shape, weight, bias)
    return normalized_shape


def _check_gradient(dy, x, shape):
    _check_tensor(dy, x.device)
    if tuple(dy.shape) != tuple(shape):
        raise ValueError("incoming gradient shape must equal the forward input shape")


def _shape(x, normalized_shape, weight, bias):
    normalized_shape = ((normalized_shape,) if isinstance(normalized_shape, int)
                        else tuple(normalized_shape))
    if not normalized_shape or len(normalized_shape) > x.ndim:
        raise ValueError("normalized_shape must describe nonempty trailing axes")
    if tuple(x.shape[-len(normalized_shape):]) != normalized_shape:
        raise ValueError("normalized_shape does not match the input trailing axes")
    if tuple(weight.shape) != normalized_shape or tuple(bias.shape) != normalized_shape:
        raise ValueError("weight and bias must have normalized_shape")
    return math.prod(x.shape[:-len(normalized_shape)]), math.prod(normalized_shape)


class LayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps):
        normalized_shape = _check_inputs(x, normalized_shape, weight, bias, eps)
        M, N = _shape(x, normalized_shape, weight, bias)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        with torch.cuda.device(x.device):
            x_arg = x.contiguous().view(M, N)
            w_arg = weight.contiguous().view(N)
            b_arg = bias.contiguous().view(N)
            y = torch.empty_like(x_arg)
            rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
            mean = torch.empty((M,), dtype=torch.float32, device=x.device)
            if M != 0 and N != 0:
                _layer_norm_fwd_fused[(M,)](
                    x_arg, y, w_arg, b_arg, mean, rstd, x_arg.stride(0), N, eps,
                    BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
            ctx.save_for_backward(x_arg, w_arg, b_arg, mean, rstd)
            ctx.BLOCK_SIZE, ctx.num_warps, ctx.eps = BLOCK_SIZE, num_warps, eps
            ctx.input_shape = x.shape
            ctx.weight_shape, ctx.weight_device = weight.shape, weight.device
            ctx.bias_shape, ctx.bias_device = bias.shape, bias.device
            return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x, w, b, mean, rstd = ctx.saved_tensors
        _check_gradient(dy, x, ctx.input_shape)
        M, N = x.shape
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        with torch.cuda.device(x.device):
            dy_arg = dy.contiguous().view(M, N)
            dx = torch.empty_like(x)
            # Empty leading dimensions require zero affine gradients.
            dw = torch.zeros_like(w)
            db = torch.zeros_like(b)
            if M != 0 and N != 0:
                groups = min(GROUP_SIZE_M, M)
                partial_dw = torch.empty((groups, N), dtype=x.dtype, device=x.device)
                partial_db = torch.empty((groups, N), dtype=x.dtype, device=x.device)
                _layer_norm_bwd_dx_fused[(groups,)](
                    dx, dy_arg, partial_dw, partial_db, x, w, mean, rstd,
                    x.stride(0), M, N, GROUP_SIZE_M=GROUP_SIZE_M,
                    BLOCK_SIZE_N=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
                _layer_norm_bwd_dwdb[(triton.cdiv(N, 128),)](
                    partial_dw, partial_db, dw, db, groups, N,
                    BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1)
            return (dx.view(ctx.input_shape), None,
                    dw.view(ctx.weight_shape),
                    db.view(ctx.bias_shape), None)


layer_norm = LayerNorm.apply

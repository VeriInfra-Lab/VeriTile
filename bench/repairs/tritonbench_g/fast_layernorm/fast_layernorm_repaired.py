"""Separate full forward/backward repair of TritonBench fast_layernorm.

Original source SHA256: 88893ebaf54a50b2b6d06841628e5a00a27f8200b026740ed6ad536deb83f6ef.
The proof profile is default FP32; explicit FP32 loads/statistics/arithmetic
and stores to the actual output dtype remain visible. No GPU validation is
claimed. Runtime grid compatibility/segmentation expansion is deferred.
"""
import triton
import triton.language as tl
import torch
import math

next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE = 65536


def calculate_settings(n: int):
    BLOCK_SIZE = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def layernorm_forward(Y, Y_row_stride, X, X_row_stride, W, b, r, mu,
                      n_cols, eps, BLOCK_SIZE: tl.constexpr, HAS_BIAS: tl.constexpr):
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols
    Y += row_idx * tl.cast(Y_row_stride, tl.int64)
    X += row_idx * tl.cast(X_row_stride, tl.int64)
    r += row_idx
    mu += row_idx
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    if HAS_BIAS:
        b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)
    else:
        b_row = tl.full((BLOCK_SIZE,), 0, tl.float32)
    mean_X = tl.sum(X_row, axis=0) / n_cols
    XX = tl.where(mask, X_row - mean_X, 0.0)
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask=mask)


@triton.jit
def layernorm_backward(dX, dX_row_stride, dY, dY_row_stride, X, X_row_stride,
                       W, r, mu, n_cols, eps, BLOCK_SIZE: tl.constexpr):
    n_cols = tl.cast(n_cols, tl.int64)
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_cols
    dX += row_idx * tl.cast(dX_row_stride, tl.int64)
    dY += row_idx * tl.cast(dY_row_stride, tl.int64)
    X += row_idx * tl.cast(X_row_stride, tl.int64)
    r += row_idx
    mu += row_idx
    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    inv_var = tl.load(r).to(tl.float32)
    mean = tl.load(mu).to(tl.float32)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    # Selecting the product prevents padded 0*NaN/Inf from entering the sum.
    product = tl.where(mask, dY_W * normed, 0.0)
    dX_row = dY_W - tl.sum(dY_W, axis=0) / n_cols - normed * tl.sum(product, axis=0) / n_cols
    dX_row = dX_row * inv_var
    tl.store(dX + col_offsets, dX_row, mask=mask)


@triton.jit
def layernorm_affine_backward(dW, db, dY, X, r, mu, n_rows, n_cols,
                              weight_count, bias_count, BLOCK_SIZE: tl.constexpr,
                              HAS_BIAS: tl.constexpr):
    n_rows = tl.cast(n_rows, tl.int64)
    n_cols = tl.cast(n_cols, tl.int64)
    weight_count = tl.cast(weight_count, tl.int64)
    bias_count = tl.cast(bias_count, tl.int64)
    columns = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    active = columns < n_cols
    dw = tl.full((BLOCK_SIZE,), 0, tl.float32)
    dbias = tl.full((BLOCK_SIZE,), 0, tl.float32)
    row = tl.full((), 0, tl.int64)
    while row < tl.cast(n_rows, tl.int64):
        offsets = row * tl.cast(n_cols, tl.int64) + tl.where(active, columns, 0)
        dy = tl.load(dY + offsets, mask=active, other=0).to(tl.float32)
        x = tl.load(X + offsets, mask=active, other=0).to(tl.float32)
        inv = tl.load(r + row).to(tl.float32)
        avg = tl.load(mu + row).to(tl.float32)
        z = (x - avg) * inv
        # All requested gradient cells are written, including unused affine tails.
        dw += tl.where(active, dy * z, 0.0)
        dbias += dy
        row += tl.full((), 1, tl.int64)
    tl.store(dW + columns, dw, mask=columns < weight_count)
    if HAS_BIAS:
        tl.store(db + columns, dbias, mask=columns < bias_count)


def _checked_tensor(tensor, device=None, contiguous=True):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a Tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError("expected a strided FP32 Tensor")
    if tensor.device.type != "cuda" or (device is not None and tensor.device != device):
        raise ValueError("expected a CUDA Tensor on the input device")
    if contiguous and not tensor.is_contiguous():
        raise ValueError("forward inputs must be contiguous")
    if tensor.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if tensor.numel() and tensor.data_ptr() % 4:
        raise ValueError("FP32 data pointer must be four-byte aligned")


def _dense_on(tensor, device):
    # Forward admission guarantees this is an identity operation. Backward
    # makes expanded/noncontiguous dY dense before reading it into separate dX.
    _checked_tensor(tensor, device=device, contiguous=False)
    return tensor.contiguous()


class Fast_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps):
        _checked_tensor(X)
        _checked_tensor(W, X.device)
        if b is not None:
            _checked_tensor(b, X.device)
        if X.ndim == 0 or X.shape[-1] == 0:
            raise ValueError("input must have a positive row width")
        try:
            valid_eps = isinstance(eps, (int, float)) and math.isfinite(eps) and eps > 0
        except (OverflowError, TypeError, ValueError):
            valid_eps = False
        if not valid_eps:
            raise ValueError("epsilon must be a finite positive Python scalar")
        shape = X.shape
        dim = shape[-1]
        n_rows, n_cols = X.reshape(-1, dim).shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        if W.numel() < n_cols or (b is not None and b.numel() < n_cols):
            raise ValueError("affine storage has fewer logical elements than the row width")
        with torch.cuda.device(X.device):
            x = _dense_on(X, X.device).view(n_rows, n_cols)
            w = _dense_on(W, X.device).view(-1)
            bias = _dense_on(b, X.device).view(-1) if b is not None else None
            Y = torch.empty_like(x)
            r = torch.empty(n_rows, dtype=torch.float32, device=X.device)
            mu = torch.empty(n_rows, dtype=torch.float32, device=X.device)
            layernorm_forward[(n_rows,)](
                Y, n_cols, x, n_cols, w,
                bias if bias is not None else w, r, mu, n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE, HAS_BIAS=bias is not None, num_warps=num_warps)
            ctx.eps = eps
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.input_shape = shape
            ctx.weight_shape, ctx.weight_device = W.shape, W.device
            ctx.bias_shape = b.shape if b is not None else None
            ctx.bias_device = b.device if b is not None else None
            ctx.save_for_backward(x, w, bias, r, mu)
            return Y.view(shape)

    @staticmethod
    def backward(ctx, dY):
        X, W, b, r, mu = ctx.saved_tensors
        n_rows, n_cols = X.shape
        _checked_tensor(dY, X.device, contiguous=False)
        if dY.shape != ctx.input_shape:
            raise ValueError("upstream gradient must have the forward output shape")
        with torch.cuda.device(X.device):
            dy = _dense_on(dY, X.device).view(n_rows, n_cols)
            dx = torch.empty_like(X)
            dw = torch.empty_like(W)
            db = torch.empty_like(b) if b is not None else None
            layernorm_backward[(n_rows,)](
                dx, n_cols, dy, n_cols, X, n_cols, W, r, mu,
                n_cols, ctx.eps, BLOCK_SIZE=ctx.BLOCK_SIZE, num_warps=ctx.num_warps)
            weight_count, bias_count = W.numel(), b.numel() if b is not None else 0
            layernorm_affine_backward[(triton.cdiv(max(weight_count, bias_count), 128),)](
                dw, db if db is not None else dw, dy, X, r, mu,
                n_rows, n_cols, weight_count, bias_count,
                BLOCK_SIZE=128, HAS_BIAS=b is not None, num_warps=4)
            return (dx.view(ctx.input_shape), dw.view(ctx.weight_shape),
                    db.view(ctx.bias_shape) if db is not None else None, None)


def fast_layernorm(layernorm, X):
    assert layernorm.elementwise_affine is True
    eps = layernorm.variance_epsilon if hasattr(layernorm, "variance_epsilon") else layernorm.eps
    return Fast_Layernorm.apply(X, layernorm.weight, layernorm.bias, eps)
